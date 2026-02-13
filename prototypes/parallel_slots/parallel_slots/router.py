"""Router orchestration and run artifact handling."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from typing import Any
from uuid import uuid4

from .config import Settings
from .openai_client import ApiCallMetric, OpenAIParallelSlotsClient
from .schemas import make_error_slot_output, validate_plan


_SLOT_ANSWER_MAX_LENGTH = 8000
_SEMANTIC_NEAR_LIMIT_RATIO = 0.95
_KNOWN_DEGENERATE_TOKENS = ("textcodelike",)


@dataclass
class SlotRunRecord:
    slot_index: int
    slot_id: str
    title: str
    output: dict[str, Any]
    api_metrics: list[ApiCallMetric]
    duration_ms: float
    slot_context: dict[str, Any]
    qc: dict[str, Any]


@dataclass
class SlotsRunResult:
    run_id: str
    run_dir: Path
    plan: dict[str, Any]
    slot_outputs: list[dict[str, Any]]
    final_text: str
    final_payload: dict[str, Any]
    metrics: dict[str, Any]


@dataclass
class BaselineRunResult:
    run_id: str
    run_dir: Path
    answer: str
    payload: dict[str, Any]
    metrics: dict[str, Any]


class ParallelSlotsRouter:
    """Deterministic router: planner -> dependency-aware workers -> ordered concatenation."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: OpenAIParallelSlotsClient,
        runs_root: Path | str = "runs",
    ):
        self._settings = settings
        self._client = client
        self._runs_root = Path(runs_root)

    def run_slots(self, user_input: str, run_id: str | None = None) -> SlotsRunResult:
        run_id = run_id or self._build_run_id(mode="slots")
        run_dir = self._runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        started_at = datetime.now(timezone.utc)
        run_start = time.perf_counter()

        plan, planner_metrics = self._client.call_planner_with_metrics(user_input)
        validate_plan(plan)

        self._validate_slot_dependencies(plan["slots"])

        self._write_json(run_dir / "plan.json", plan)

        slots = plan["slots"]
        slot_total = len(slots)
        slot_by_id = {slot["slot_id"]: slot for slot in slots}
        slot_index_by_id = {slot["slot_id"]: idx for idx, slot in enumerate(slots, start=1)}

        completed_slot_ids: set[str] = set()
        pending_slot_ids = [slot["slot_id"] for slot in slots]
        execution_waves: list[list[str]] = []
        peak_worker_count = 0
        slot_records_by_index: dict[int, SlotRunRecord] = {}

        while pending_slot_ids:
            ready_slot_ids = [
                slot_id
                for slot_id in pending_slot_ids
                if all(dep_id in completed_slot_ids for dep_id in slot_by_id[slot_id]["depends_on"])
            ]

            if not ready_slot_ids:
                for slot_id in pending_slot_ids:
                    slot = slot_by_id[slot_id]
                    slot_context = self._build_slot_context(slot)
                    unresolved = [dep for dep in slot["depends_on"] if dep not in completed_slot_ids]
                    slot_output = make_error_slot_output(
                        slot_id=slot["slot_id"],
                        title=slot["title"],
                        reason=f"Dependency unresolved: {', '.join(unresolved)}",
                    )
                    slot_output["needs_tools"] = list(slot_context["missing_required"])
                    slot_records_by_index[slot_index_by_id[slot_id]] = SlotRunRecord(
                        slot_index=slot_index_by_id[slot_id],
                        slot_id=slot["slot_id"],
                        title=slot["title"],
                        output=slot_output,
                        api_metrics=[],
                        duration_ms=0.0,
                        slot_context=slot_context,
                        qc={
                            "semantic_retry_used": False,
                            "semantic_status": "skipped_due_dependency_error",
                            "semantic_fail_reasons": [],
                        },
                    )
                break

            execution_waves.append(ready_slot_ids.copy())
            worker_count = min(self._settings.max_concurrency, len(ready_slot_ids))
            peak_worker_count = max(peak_worker_count, worker_count)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {}
                for slot_id in ready_slot_ids:
                    slot = slot_by_id[slot_id]
                    slot_index = slot_index_by_id[slot_id]
                    slot_context = self._build_slot_context(slot)
                    future = executor.submit(
                        self._run_single_slot,
                        user_input,
                        plan,
                        slot,
                        slot_index,
                        slot_total,
                        slot_context,
                    )
                    futures[future] = slot_index

                for future in as_completed(futures):
                    slot_index = futures[future]
                    slot_records_by_index[slot_index] = future.result()

            completed_slot_ids.update(ready_slot_ids)
            pending_slot_ids = [slot_id for slot_id in pending_slot_ids if slot_id not in completed_slot_ids]

        slot_records = [slot_records_by_index[index] for index in range(1, slot_total + 1)]
        slot_outputs = [record.output for record in slot_records]

        for record in slot_records:
            self._write_json(run_dir / f"slot_{record.slot_id}.json", record.output)

        final_text = concat_slot_answers(slot_outputs)
        final_payload = {
            "run_id": run_id,
            "mode": "slots",
            "user_input": user_input,
            "plan": plan,
            "slots": slot_outputs,
            "execution_waves": execution_waves,
        }

        finished_at = datetime.now(timezone.utc)
        total_duration_ms = (time.perf_counter() - run_start) * 1000

        all_api_metrics = planner_metrics.copy()
        for record in slot_records:
            all_api_metrics.extend(record.api_metrics)

        metrics = {
            "run_id": run_id,
            "mode": "slots",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "total_duration_ms": round(total_duration_ms, 2),
            "configured_max_concurrency": self._settings.max_concurrency,
            "effective_concurrency": peak_worker_count,
            "slot_total": slot_total,
            "execution_waves": execution_waves,
            "api_calls": [metric.to_dict() for metric in all_api_metrics],
            "slot_runs": [
                {
                    "slot_id": record.slot_id,
                    "slot_index": record.slot_index,
                    "title": record.title,
                    "status": record.output.get("status"),
                    "duration_ms": round(record.duration_ms, 2),
                    "missing_required": record.slot_context.get("missing_required", []),
                    "injected_vars": sorted(record.slot_context.get("injected", {}).keys()),
                    "qc": record.qc,
                    "api_calls": [metric.to_dict() for metric in record.api_metrics],
                }
                for record in slot_records
            ],
            "token_usage": _sum_tokens(all_api_metrics),
        }

        final_payload["metrics"] = metrics

        self._write_json(run_dir / "final.json", final_payload)
        (run_dir / "final.txt").write_text(final_text, encoding="utf-8")
        self._write_json(run_dir / "metrics.json", metrics)

        return SlotsRunResult(
            run_id=run_id,
            run_dir=run_dir,
            plan=plan,
            slot_outputs=slot_outputs,
            final_text=final_text,
            final_payload=final_payload,
            metrics=metrics,
        )

    def run_baseline(self, user_input: str, run_id: str | None = None) -> BaselineRunResult:
        run_id = run_id or self._build_run_id(mode="baseline")
        run_dir = self._runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        started_at = datetime.now(timezone.utc)
        run_start = time.perf_counter()

        answer, api_metrics = self._client.call_baseline_with_metrics(user_input)

        finished_at = datetime.now(timezone.utc)
        total_duration_ms = (time.perf_counter() - run_start) * 1000

        metrics = {
            "run_id": run_id,
            "mode": "baseline",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "total_duration_ms": round(total_duration_ms, 2),
            "api_calls": [metric.to_dict() for metric in api_metrics],
            "token_usage": _sum_tokens(api_metrics),
        }

        payload = {
            "run_id": run_id,
            "mode": "baseline",
            "user_input": user_input,
            "answer": answer,
            "metrics": metrics,
        }

        (run_dir / "baseline_answer.txt").write_text(answer, encoding="utf-8")
        self._write_json(run_dir / "baseline_final.json", payload)
        self._write_json(run_dir / "baseline_metrics.json", metrics)

        return BaselineRunResult(
            run_id=run_id,
            run_dir=run_dir,
            answer=answer,
            payload=payload,
            metrics=metrics,
        )

    def _run_single_slot(
        self,
        user_input: str,
        plan: dict[str, Any],
        slot: dict[str, Any],
        slot_index: int,
        slot_total: int,
        slot_context: dict[str, Any],
    ) -> SlotRunRecord:
        slot_start = time.perf_counter()
        qc_meta = {
            "semantic_retry_used": False,
            "semantic_status": "pass",
            "semantic_fail_reasons": [],
        }

        try:
            missing_required = slot_context.get("missing_required", [])
            if slot["risk"] == "high" and missing_required:
                slot_output = make_error_slot_output(
                    slot_id=slot["slot_id"],
                    title=slot["title"],
                    reason=f"High-risk slot blocked: missing required tools {missing_required}",
                )
                slot_output["needs_tools"] = list(missing_required)
                qc_meta["semantic_status"] = "skipped_high_risk_missing_tools"
                return self._build_slot_record(
                    slot=slot,
                    slot_index=slot_index,
                    slot_context=slot_context,
                    output=slot_output,
                    api_metrics=[],
                    duration_ms=(time.perf_counter() - slot_start) * 1000,
                    qc_meta=qc_meta,
                )

            slot_output, api_metrics = self._client.call_worker_with_metrics(
                user_input=user_input,
                plan=plan,
                slot=slot,
                slot_index=slot_index,
                slot_total=slot_total,
                slot_context=slot_context,
            )

            if slot_output.get("status") == "ok":
                degradation_reasons = _detect_semantic_degradation(
                    answer=slot_output.get("answer", ""),
                    expected_hint=slot.get("expected_output_schema_hint", ""),
                )
                if degradation_reasons:
                    qc_meta["semantic_retry_used"] = True
                    qc_meta["semantic_status"] = "retry_triggered"
                    qc_meta["semantic_fail_reasons"] = degradation_reasons

                    retry_output, retry_metrics = self._client.call_worker_with_metrics(
                        user_input=user_input,
                        plan=plan,
                        slot=slot,
                        slot_index=slot_index,
                        slot_total=slot_total,
                        slot_context=slot_context,
                        quality_guardrails=self._build_qc_retry_guardrails(slot, degradation_reasons),
                        temperature_override=self._build_qc_retry_temperature(),
                    )
                    api_metrics.extend(retry_metrics)

                    retry_reasons = []
                    if retry_output.get("status") == "ok":
                        retry_reasons = _detect_semantic_degradation(
                            answer=retry_output.get("answer", ""),
                            expected_hint=slot.get("expected_output_schema_hint", ""),
                        )
                    else:
                        retry_reasons = ["retry_status_not_ok"]

                    if retry_output.get("status") == "ok" and not retry_reasons:
                        slot_output = retry_output
                        qc_meta["semantic_status"] = "recovered_after_retry"
                        qc_meta["semantic_fail_reasons"] = []
                    else:
                        qc_meta["semantic_status"] = "failed_after_retry"
                        qc_meta["semantic_fail_reasons"] = retry_reasons
                        slot_output = self._build_semantic_error_output(
                            slot=slot,
                            slot_context=slot_context,
                            original_output=slot_output,
                            initial_reasons=degradation_reasons,
                            retry_output=retry_output,
                            retry_reasons=retry_reasons,
                        )
        except Exception as exc:
            slot_output = make_error_slot_output(slot["slot_id"], slot["title"], f"Worker crashed: {exc}")
            api_metrics = []
            qc_meta["semantic_status"] = "worker_exception"

        duration_ms = (time.perf_counter() - slot_start) * 1000
        return self._build_slot_record(
            slot=slot,
            slot_index=slot_index,
            slot_context=slot_context,
            output=slot_output,
            api_metrics=api_metrics,
            duration_ms=duration_ms,
            qc_meta=qc_meta,
        )

    @staticmethod
    def _build_slot_record(
        *,
        slot: dict[str, Any],
        slot_index: int,
        slot_context: dict[str, Any],
        output: dict[str, Any],
        api_metrics: list[ApiCallMetric],
        duration_ms: float,
        qc_meta: dict[str, Any],
    ) -> SlotRunRecord:
        return SlotRunRecord(
            slot_index=slot_index,
            slot_id=slot["slot_id"],
            title=slot["title"],
            output=output,
            api_metrics=api_metrics,
            duration_ms=duration_ms,
            slot_context=slot_context,
            qc=qc_meta,
        )

    @staticmethod
    def _build_slot_context(slot: dict[str, Any]) -> dict[str, Any]:
        injected: dict[str, dict[str, Any]] = {}
        missing_required: list[str] = []

        for request in slot.get("tool_requests", []):
            bind_var = str(request.get("bind_var", "")).strip()
            if not bind_var:
                continue
            # Tool execution is intentionally not enabled in this prototype stage.
            if bool(request.get("required", False)):
                missing_required.append(bind_var)

        return {
            "injected": injected,
            "missing_required": sorted(set(missing_required)),
            "injected_format": "evidence_pack.v1",
            "evidence_pack_template": _evidence_pack_template(),
        }

    def _build_semantic_error_output(
        self,
        *,
        slot: dict[str, Any],
        slot_context: dict[str, Any],
        original_output: dict[str, Any],
        initial_reasons: list[str],
        retry_output: dict[str, Any],
        retry_reasons: list[str],
    ) -> dict[str, Any]:
        raw_answer = str(original_output.get("answer", ""))
        retry_answer = str(retry_output.get("answer", ""))

        reason = (
            "Semantic health check failed after one retry. "
            f"initial_reasons={initial_reasons}; retry_reasons={retry_reasons}; "
            f"raw_output_excerpt={_truncate_text(raw_answer)}; "
            f"retry_output_excerpt={_truncate_text(retry_answer)}"
        )
        error_output = make_error_slot_output(slot["slot_id"], slot["title"], reason)
        error_output["answer"] = raw_answer
        error_output["evidence_used"] = [str(item) for item in original_output.get("evidence_used", [])]
        error_output["unsupported_claims"] = [
            str(item) for item in original_output.get("unsupported_claims", [])
        ]
        needs_tools = {str(item) for item in original_output.get("needs_tools", [])}
        needs_tools.update(str(item) for item in slot_context.get("missing_required", []))
        error_output["needs_tools"] = sorted(needs_tools)
        return error_output

    def _build_qc_retry_temperature(self) -> float:
        return max(0.0, min(2.0, self._settings.worker_temperature * 0.5))

    @staticmethod
    def _build_qc_retry_guardrails(slot: dict[str, Any], reasons: list[str]) -> str:
        base_rules = [
            "Regenerate with coherent natural language and avoid repetitive characters/phrases.",
            "You must provide at least 3 concrete key points as markdown bullets.",
            "Every key claim should either cite evidence_used or appear in unsupported_claims.",
        ]
        intent = f"{slot.get('worker_brief', '')} {slot.get('expected_output_schema_hint', '')}".lower()
        if any(token in intent for token in ("code", "代码", "sql", "script", "脚本")):
            base_rules.append("Include at least one fenced code block.")
        return f"degradation_reasons={reasons}\n" + "\n".join(base_rules)

    @staticmethod
    def _validate_slot_dependencies(slots: list[dict[str, Any]]) -> None:
        slot_ids = [slot["slot_id"] for slot in slots]
        slot_id_set = set(slot_ids)
        if len(slot_ids) != len(slot_id_set):
            raise ValueError("plan slots contain duplicate slot_id")

        for slot in slots:
            slot_id = slot["slot_id"]
            for dep_id in slot.get("depends_on", []):
                if dep_id == slot_id:
                    raise ValueError(f"slot {slot_id} cannot depend on itself")
                if dep_id not in slot_id_set:
                    raise ValueError(f"slot {slot_id} depends on unknown slot_id {dep_id}")

    @staticmethod
    def _build_run_id(mode: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = uuid4().hex[:8]
        return f"{mode}_{timestamp}_{suffix}"

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def concat_slot_answers(slot_outputs: list[dict[str, Any]]) -> str:
    """Concatenate slot answers in slot order without semantic merge."""

    sections: list[str] = []
    for slot_output in slot_outputs:
        slot_id = slot_output.get("slot_id", "unknown")
        title = slot_output.get("title", "untitled")
        answer = slot_output.get("answer", "")
        sections.append(f"[{slot_id}] {title}\n\n{answer}".strip())
    return "\n\n-----\n\n".join(sections)


def _detect_semantic_degradation(answer: str, expected_hint: str) -> list[str]:
    reasons: list[str] = []
    normalized = answer or ""
    normalized_lower = normalized.lower()

    if len(normalized) >= int(_SLOT_ANSWER_MAX_LENGTH * _SEMANTIC_NEAR_LIMIT_RATIO):
        reasons.append("near_max_length_limit")

    if any(token in normalized_lower for token in _KNOWN_DEGENERATE_TOKENS):
        reasons.append("known_degenerate_token")

    compact = "".join(ch for ch in normalized if not ch.isspace())
    if len(compact) >= 80:
        char_counts = Counter(compact)
        max_ratio = max(char_counts.values()) / len(compact)
        if max_ratio >= 0.35:
            reasons.append("abnormal_char_distribution")

    if len(compact) >= 80 and _char_entropy(compact) < 2.0:
        reasons.append("low_entropy_text")

    if len(compact) >= 80 and re.search(r"(.{2,20})\1{4,}", compact):
        reasons.append("long_repeated_pattern")

    hint_lower = (expected_hint or "").lower()
    promised_list_or_table = any(
        token in hint_lower or token in normalized_lower
        for token in ("列表", "表格", "list", "table", "要点", "步骤")
    )
    has_sentence_break = any(token in normalized for token in ("。", ".", "!", "?", "\n"))
    if promised_list_or_table and not has_sentence_break:
        reasons.append("missing_sentence_break_for_promised_structure")

    return reasons


def _char_entropy(text: str) -> float:
    total = len(text)
    if total == 0:
        return 0.0
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _truncate_text(value: str, max_chars: int = 800) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...(truncated)"


def _evidence_pack_template() -> dict[str, Any]:
    return {
        "items": [
            {
                "source_type": "example",
                "source_id": "example-id",
                "title": "example-title",
                "snippet": "example-snippet",
                "meta": {},
            }
        ]
    }


def _sum_tokens(metrics: list[ApiCallMetric]) -> dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for metric in metrics:
        if metric.input_tokens is not None:
            input_tokens += metric.input_tokens
        if metric.output_tokens is not None:
            output_tokens += metric.output_tokens
        if metric.total_tokens is not None:
            total_tokens += metric.total_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
