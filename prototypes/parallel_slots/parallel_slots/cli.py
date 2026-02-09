"""CLI entrypoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, pvariance
import sys
from typing import Any

from .config import ConfigError, load_settings
from .openai_client import OpenAIParallelSlotsClient, PlannerExecutionError
from .router import BaselineRunResult, ParallelSlotsRouter, SlotsRunResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel Slots Prototype CLI")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Inline user input text")
    input_group.add_argument("--input-file", help="Read user input from file")
    parser.add_argument("--mode", choices=["slots", "baseline"], default="slots")
    parser.add_argument("--repeat", type=int, default=1, help="Run the same mode N times")
    parser.add_argument("--runs-root", default="runs", help="Directory for run artifacts (default: runs)")
    return parser


def _resolve_user_input(args: argparse.Namespace) -> str:
    if args.input is not None:
        return args.input.strip()
    input_file = Path(args.input_file)
    return input_file.read_text(encoding="utf-8").strip()


def _pretty_print(label: str, payload: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_slots_result(result: SlotsRunResult) -> None:
    _pretty_print("Plan", result.plan)
    for slot_output in result.slot_outputs:
        slot_id = slot_output.get("slot_id", "unknown")
        _pretty_print(f"Slot {slot_id}", slot_output)

    print("\n=== Final Text ===")
    print(result.final_text)
    print("\n=== Run Directory ===")
    print(str(result.run_dir))


def _print_baseline_result(result: BaselineRunResult) -> None:
    print("\n=== Baseline Answer ===")
    print(result.answer)
    _pretty_print("Baseline Metrics", result.metrics)
    print("\n=== Run Directory ===")
    print(str(result.run_dir))


def _build_repeat_summary(mode: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [run["total_duration_ms"] for run in runs]
    total_tokens = [run["token_usage"].get("total_tokens", 0) for run in runs]

    duration_variance = pvariance(durations) if len(durations) > 1 else 0.0
    token_variance = pvariance(total_tokens) if len(total_tokens) > 1 else 0.0

    return {
        "mode": mode,
        "repeat": len(runs),
        "runs": runs,
        "aggregates": {
            "duration_ms_mean": round(mean(durations), 4),
            "duration_ms_variance": round(duration_variance, 4),
            "total_tokens_mean": round(mean(total_tokens), 4),
            "total_tokens_variance": round(token_variance, 4),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.repeat < 1:
            raise ValueError("--repeat must be >= 1")

        user_input = _resolve_user_input(args)
        if not user_input:
            raise ValueError("Input cannot be empty.")

        settings = load_settings()
        runs_root = Path(args.runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)

        client = OpenAIParallelSlotsClient(settings)
        router = ParallelSlotsRouter(settings=settings, client=client, runs_root=runs_root)

        run_rows: list[dict[str, Any]] = []

        for run_idx in range(1, args.repeat + 1):
            if args.repeat > 1:
                print(f"\n##### Run {run_idx}/{args.repeat} ({args.mode}) #####")

            if args.mode == "slots":
                result = router.run_slots(user_input)
                _print_slots_result(result)
                metrics = result.metrics
                run_rows.append(
                    {
                        "run_id": result.run_id,
                        "run_dir": str(result.run_dir),
                        "total_duration_ms": metrics["total_duration_ms"],
                        "token_usage": metrics.get("token_usage", {}),
                    }
                )
            else:
                result = router.run_baseline(user_input)
                _print_baseline_result(result)
                metrics = result.metrics
                run_rows.append(
                    {
                        "run_id": result.run_id,
                        "run_dir": str(result.run_dir),
                        "total_duration_ms": metrics["total_duration_ms"],
                        "token_usage": metrics.get("token_usage", {}),
                    }
                )

        summary = _build_repeat_summary(args.mode, run_rows)
        summary_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        summary_path = runs_root / f"repeat_summary_{args.mode}_{summary_ts}.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        _pretty_print("Repeat Summary", summary)
        print("\n=== Summary File ===")
        print(str(summary_path))
        return 0
    except (ConfigError, PlannerExecutionError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
