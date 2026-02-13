from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from parallel_slots.config import Settings
from parallel_slots.openai_client import ApiCallMetric
from parallel_slots.router import ParallelSlotsRouter


def _api_metric(operation: str) -> ApiCallMetric:
    return ApiCallMetric(
        operation=operation,
        model="fake-worker",
        attempt=1,
        duration_ms=15.0,
        success=True,
        parse_ok=True,
        error=None,
        input_tokens=11,
        output_tokens=21,
        total_tokens=32,
    )


class FakeClient:
    def call_planner_with_metrics(self, user_input: str) -> tuple[dict[str, Any], list[ApiCallMetric]]:
        plan = {
            "version": "plan.v1",
            "plan_id": "plan-test-1",
            "language": "zh",
            "task_summary": "验证 router 是否按 slot 顺序输出",
            "tool_contract": {
                "enabled": False,
                "mode": "mock",
                "notes": "MVP 工具合约占位",
            },
            "slots": [
                {
                    "slot_id": "S1",
                    "title": "先写第一段",
                    "worker_brief": "你是第 1/2 个 slot",
                    "expected_output_schema_hint": "重点 answer",
                    "tool_requests": [],
                    "depends_on": [],
                    "budget": {"max_tokens": 800, "priority": 8},
                    "risk": "low",
                },
                {
                    "slot_id": "S2",
                    "title": "再写第二段",
                    "worker_brief": "你是第 2/2 个 slot",
                    "expected_output_schema_hint": "重点 answer",
                    "tool_requests": [],
                    "depends_on": [],
                    "budget": {"max_tokens": 800, "priority": 8},
                    "risk": "low",
                },
            ],
            "output_skeleton": "{{S1}}\n\n{{S2}}",
            "router_notes": "并发即可",
            "questions_to_user": [],
        }
        metric = _api_metric("planner")
        return plan, [metric]

    def call_worker_with_metrics(
        self,
        user_input: str,
        plan: dict[str, Any],
        slot: dict[str, Any],
        slot_index: int,
        slot_total: int,
        slot_context: dict[str, Any],
        quality_guardrails: str | None = None,
        temperature_override: float | None = None,
    ) -> tuple[dict[str, Any], list[ApiCallMetric]]:
        # Make S1 slower than S2 to ensure completion order differs from slot order.
        if slot["slot_id"] == "S1":
            time.sleep(0.05)
        else:
            time.sleep(0.01)

        output = {
            "version": "slot_output.v1",
            "status": "ok",
            "slot_id": slot["slot_id"],
            "title": slot["title"],
            "answer": f"answer for {slot['slot_id']}",
            "assumptions": [],
            "open_questions": [],
            "evidence_used": [],
            "unsupported_claims": [],
            "needs_tools": [],
            "confidence": 0.9,
            "notes_for_router": "",
        }

        metric = _api_metric(f"worker:{slot['slot_id']}")
        return output, [metric]


def _settings() -> Settings:
    return Settings(
        openai_api_key="test",
        openai_base_url=None,
        openai_org_id=None,
        openai_project_id=None,
        planner_model="fake-planner",
        worker_model="fake-worker",
        planner_temperature=0.2,
        worker_temperature=0.2,
        max_concurrency=4,
        max_retries=2,
        retry_base_delay_ms=10,
    )


def test_router_concatenates_in_slot_order_with_mock_client(tmp_path: Path) -> None:
    router = ParallelSlotsRouter(
        settings=_settings(),
        client=FakeClient(),
        runs_root=tmp_path / "runs",
    )

    result = router.run_slots("demo input", run_id="slots_test")

    assert [slot["slot_id"] for slot in result.slot_outputs] == ["S1", "S2"]
    assert result.final_text.index("[S1]") < result.final_text.index("[S2]")

    run_dir = tmp_path / "runs" / "slots_test"
    assert (run_dir / "plan.json").exists()
    assert (run_dir / "slot_S1.json").exists()
    assert (run_dir / "slot_S2.json").exists()
    assert (run_dir / "final.json").exists()
    assert (run_dir / "final.txt").exists()
    assert (run_dir / "metrics.json").exists()

    final_payload = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    assert [slot["slot_id"] for slot in final_payload["slots"]] == ["S1", "S2"]


def test_router_respects_dependency_waves(tmp_path: Path) -> None:
    class DependencyClient(FakeClient):
        def call_planner_with_metrics(self, user_input: str) -> tuple[dict[str, Any], list[ApiCallMetric]]:
            plan, metrics = super().call_planner_with_metrics(user_input)
            plan["slots"][1]["depends_on"] = ["S1"]
            return plan, metrics

    router = ParallelSlotsRouter(
        settings=_settings(),
        client=DependencyClient(),
        runs_root=tmp_path / "runs",
    )

    result = router.run_slots("demo input", run_id="slots_dep")
    assert result.metrics["execution_waves"] == [["S1"], ["S2"]]


def test_router_blocks_high_risk_slot_without_required_tools(tmp_path: Path) -> None:
    class HighRiskClient(FakeClient):
        def __init__(self) -> None:
            self.worker_calls = 0

        def call_planner_with_metrics(self, user_input: str) -> tuple[dict[str, Any], list[ApiCallMetric]]:
            plan, metrics = super().call_planner_with_metrics(user_input)
            plan["slots"] = [
                {
                    "slot_id": "S1",
                    "title": "高风险 slot",
                    "worker_brief": "必须依据外部证据回答",
                    "expected_output_schema_hint": "重点 answer",
                    "tool_requests": [
                        {
                            "call_id": "call_risk_fact",
                            "tool_name": "web.search",
                            "args": {"q": "latest risk facts"},
                            "bind_var": "risk_fact",
                            "required": True,
                        }
                    ],
                    "depends_on": [],
                    "budget": {"max_tokens": 256, "priority": 10},
                    "risk": "high",
                }
            ]
            plan["output_skeleton"] = "{{S1}}"
            return plan, metrics

        def call_worker_with_metrics(
            self,
            user_input: str,
            plan: dict[str, Any],
            slot: dict[str, Any],
            slot_index: int,
            slot_total: int,
            slot_context: dict[str, Any],
            quality_guardrails: str | None = None,
            temperature_override: float | None = None,
        ) -> tuple[dict[str, Any], list[ApiCallMetric]]:
            self.worker_calls += 1
            return super().call_worker_with_metrics(
                user_input,
                plan,
                slot,
                slot_index,
                slot_total,
                slot_context,
                quality_guardrails=quality_guardrails,
                temperature_override=temperature_override,
            )

    client = HighRiskClient()
    router = ParallelSlotsRouter(
        settings=_settings(),
        client=client,
        runs_root=tmp_path / "runs",
    )

    result = router.run_slots("demo input", run_id="slots_high_risk")
    assert client.worker_calls == 0
    assert result.slot_outputs[0]["status"] == "error"
    assert result.slot_outputs[0]["needs_tools"] == ["risk_fact"]


def test_router_semantic_qc_retries_then_errors(tmp_path: Path) -> None:
    class DegenerateClient(FakeClient):
        def __init__(self) -> None:
            self.worker_calls = 0

        def call_planner_with_metrics(self, user_input: str) -> tuple[dict[str, Any], list[ApiCallMetric]]:
            plan, metrics = super().call_planner_with_metrics(user_input)
            plan["slots"] = [plan["slots"][0]]
            plan["output_skeleton"] = "{{S1}}"
            return plan, metrics

        def call_worker_with_metrics(
            self,
            user_input: str,
            plan: dict[str, Any],
            slot: dict[str, Any],
            slot_index: int,
            slot_total: int,
            slot_context: dict[str, Any],
            quality_guardrails: str | None = None,
            temperature_override: float | None = None,
        ) -> tuple[dict[str, Any], list[ApiCallMetric]]:
            self.worker_calls += 1
            output = {
                "version": "slot_output.v1",
                "status": "ok",
                "slot_id": slot["slot_id"],
                "title": slot["title"],
                "answer": "textcodelike" * 300,
                "assumptions": [],
                "open_questions": [],
                "evidence_used": [],
                "unsupported_claims": [],
                "needs_tools": [],
                "confidence": 0.3,
                "notes_for_router": "",
            }
            return output, [_api_metric(f"worker:{slot['slot_id']}")]

    client = DegenerateClient()
    router = ParallelSlotsRouter(
        settings=_settings(),
        client=client,
        runs_root=tmp_path / "runs",
    )

    result = router.run_slots("demo input", run_id="slots_qc_error")
    assert client.worker_calls == 2
    assert result.slot_outputs[0]["status"] == "error"
    assert "Semantic health check failed" in result.slot_outputs[0]["notes_for_router"]
