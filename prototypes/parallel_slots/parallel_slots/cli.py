"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import ConfigError, load_settings
from .openai_client import OpenAIParallelSlotsClient, PlannerExecutionError
from .router import ParallelSlotsRouter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel Slots Prototype CLI")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Inline user input text")
    input_group.add_argument("--input-file", help="Read user input from file")
    parser.add_argument("--runs-root", default="runs", help="Directory for run artifacts (default: runs)")
    return parser


def _resolve_user_input(args: argparse.Namespace) -> str:
    if args.input is not None:
        return args.input.strip()
    input_file = Path(args.input_file)
    return input_file.read_text(encoding="utf-8").strip()


def _pretty_print(label: str, payload: dict) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        user_input = _resolve_user_input(args)
        if not user_input:
            raise ValueError("Input cannot be empty.")

        settings = load_settings()
        client = OpenAIParallelSlotsClient(settings)
        router = ParallelSlotsRouter(settings=settings, client=client, runs_root=args.runs_root)

        result = router.run_slots(user_input)

        _pretty_print("Plan", result.plan)
        for slot_output in result.slot_outputs:
            slot_id = slot_output.get("slot_id", "unknown")
            _pretty_print(f"Slot {slot_id}", slot_output)

        print("\n=== Final Text ===")
        print(result.final_text)
        print("\n=== Run Directory ===")
        print(str(result.run_dir))
        return 0
    except (ConfigError, PlannerExecutionError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
