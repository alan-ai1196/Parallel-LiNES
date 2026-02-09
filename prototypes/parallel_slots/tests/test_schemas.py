from __future__ import annotations

import json
from pathlib import Path

import pytest

from parallel_slots.schemas import make_error_slot_output, validate_plan, validate_slot_output


_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_validate_plan_fixture() -> None:
    plan = _load_fixture("plan_valid.json")
    assert validate_plan(plan) == plan


def test_validate_plan_rejects_missing_required_field() -> None:
    invalid_plan = _load_fixture("plan_valid.json")
    invalid_plan.pop("task_summary")

    with pytest.raises(ValueError):
        validate_plan(invalid_plan)


def test_validate_slot_output_fixture() -> None:
    slot_output = _load_fixture("slot_output_valid.json")
    assert validate_slot_output(slot_output) == slot_output


def test_validate_slot_output_rejects_extra_property() -> None:
    invalid_slot = _load_fixture("slot_output_valid.json")
    invalid_slot["unexpected"] = "boom"

    with pytest.raises(ValueError):
        validate_slot_output(invalid_slot)


def test_error_slot_output_matches_schema() -> None:
    error_output = make_error_slot_output("S9", "fallback", "structured failure")
    assert validate_slot_output(error_output) == error_output
