"""Router orchestration and run artifact handling."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from .config import Settings
from .openai_client import ApiCallMetric, OpenAIParallelSlotsClient
from .schemas import make_error_slot_output, validate_plan


@dataclass
class SlotRunRecord:
    slot_index: int
    slot_id: str
    title: str
    output: dict[str, Any]
    api_metrics: list[ApiCallMetric]
    duration_ms: float


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
    """Deterministic router: planner -> concurrent workers -> ordered concatenation."""

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

        plan_path = run_dir / "plan.json"
        self._write_json(plan_path, plan)

        slots = plan["slots"]
        slot_total = len(slots)
        worker_count = min(self._settings.max_concurrency, slot_total)

        slot_records_by_index: dict[int, SlotRunRecord] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    self._run_single_slot,
                    user_input,
                    plan,
                    slot,
                    slot_index,
                    slot_total,
                ): slot_index
                for slot_index, slot in enumerate(slots, start=1)
            }

            for future in as_completed(futures):
                slot_index = futures[future]
                slot_records_by_index[slot_index] = future.result()

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
            "effective_concurrency": worker_count,
            "slot_total": slot_total,
            "api_calls": [metric.to_dict() for metric in all_api_metrics],
            "slot_runs": [
                {
                    "slot_id": record.slot_id,
                    "slot_index": record.slot_index,
                    "title": record.title,
                    "status": record.output.get("status"),
                    "duration_ms": round(record.duration_ms, 2),
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
    ) -> SlotRunRecord:
        slot_start = time.perf_counter()

        try:
            slot_output, api_metrics = self._client.call_worker_with_metrics(
                user_input=user_input,
                plan=plan,
                slot=slot,
                slot_index=slot_index,
                slot_total=slot_total,
            )
        except Exception as exc:
            slot_output = make_error_slot_output(slot["slot_id"], slot["title"], f"Worker crashed: {exc}")
            api_metrics = []

        duration_ms = (time.perf_counter() - slot_start) * 1000
        return SlotRunRecord(
            slot_index=slot_index,
            slot_id=slot["slot_id"],
            title=slot["title"],
            output=slot_output,
            api_metrics=api_metrics,
            duration_ms=duration_ms,
        )

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
