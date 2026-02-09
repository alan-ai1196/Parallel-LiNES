from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from parallel_slots.config import Settings
from parallel_slots.openai_client import ApiCallMetric
from parallel_slots.router import ParallelSlotsRouter


class FakeClient:
    def call_planner_with_metrics(self, user_input: str) -> tuple[dict[str, Any], list[ApiCallMetric]]:
        plan = {
            "version": "plan.v1",
            "plan_id": "plan-test-1",
            "language": "zh",
            "task_summary": "验证 router 是否按 slot 顺序输出",
            "slots": [
                {
                    "slot_id": "S1",
                    "title": "先写第一段",
                    "worker_brief": "你是第 1/2 个 slot",
                    "expected_output_schema_hint": "重点 answer",
                },
                {
                    "slot_id": "S2",
                    "title": "再写第二段",
                    "worker_brief": "你是第 2/2 个 slot",
                    "expected_output_schema_hint": "重点 answer",
                },
            ],
            "router_notes": "并发即可",
            "questions_to_user": [],
        }
        metric = ApiCallMetric(
            operation="planner",
            model="fake-planner",
            attempt=1,
            duration_ms=5.0,
            success=True,
            parse_ok=True,
            error=None,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )
        return plan, [metric]

    def call_worker_with_metrics(
        self,
        user_input: str,
        plan: dict[str, Any],
        slot: dict[str, Any],
        slot_index: int,
        slot_total: int,
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
            "confidence": 0.9,
            "notes_for_router": "",
        }

        metric = ApiCallMetric(
            operation=f"worker:{slot['slot_id']}",
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
