"""Run a tiny Qwen3 repository smoke suite and persist full trajectories.

This is a Phase A integration check, not a benchmark. Every task runs in a
disposable copy of the repository; the safe-change task cannot modify the
source checkout. Model loading is local-only and never downloads weights.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent.loop import AgentLoop, AgentResult
from agent.state import AgentConfig, StopReason
from evals.manual_tasks import MANUAL_SMOKE_TASKS
from model.local import LocalTransformersClient
from tools import Toolbox
from tools.executor import ToolExecutor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COPY_IGNORES = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".terminal-slm", "models", "checkpoints", "results", "raw",
    ".aws", ".ssh",
}
COPY_FILE_PATTERNS = ("*.pyc", "*.pyo", "*.gguf", "*.safetensors", "*.pt", "*.pth")


def _ignore_copy_entries(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        candidate = Path(directory) / name
        if name in COPY_IGNORES or name.startswith(".env") or candidate.is_symlink():
            ignored.add(name)
        elif candidate.is_file() and any(fnmatch.fnmatch(name, pattern) for pattern in COPY_FILE_PATTERNS):
            ignored.add(name)
    return ignored


def _tool_metrics(result: AgentResult) -> dict[str, Any]:
    events = result.trajectory.events
    tool_events = [event for event in events if event.kind == "tool"]
    model_events = [event for event in events if event.kind == "model"]
    got_model_output = any(event.model_output is not None for event in model_events)
    emitted_native_blocks = (
        sum((event.model_output or "").count("<tool_call>") for event in model_events)
        if got_model_output
        else None
    )
    valid_tool_calls = sum(event.tool_name is not None for event in tool_events)
    executed = [event for event in tool_events if event.executed]
    return {
        "model_turns": result.turns,
        "native_tool_blocks_emitted": emitted_native_blocks,
        "valid_tool_calls": valid_tool_calls if got_model_output else None,
        "tool_call_assessment": "measured" if got_model_output else "not_reached_model_inference_unavailable",
        "invalid_tool_call_responses": int(result.stop_reason.value == "invalid_tool_call"),
        "tool_calls_executed": len(executed),
        "tool_executions_succeeded": sum(
            bool(event.tool_result and event.tool_result.get("success")) for event in executed
        ),
        "tool_steps": len(tool_events),
    }


def _diagnose(result: AgentResult) -> dict[str, str]:
    if result.stop_reason == StopReason.INVALID_TOOL_CALL:
        return {
            "category": "parser_or_native_tool_call_format",
            "detail": next(
                (event.error for event in result.trajectory.events if event.error),
                "assistant tool-call output was rejected",
            ) or "assistant tool-call output was rejected",
            "downstream": "tool_execution_not_started_for_invalid_call",
        }
    model_errors = [event.error for event in result.trajectory.events if event.kind == "model" and event.error]
    if model_errors:
        return {
            "category": "model_inference",
            "detail": str(model_errors[0]),
            "downstream": "not_assessed_no_model_response",
        }
    tool_errors = [
        event.error or str((event.tool_result or {}).get("error") or "tool returned failure")
        for event in result.trajectory.events
        if event.kind == "tool" and event.executed and not (event.tool_result or {}).get("success")
    ]
    if tool_errors:
        return {"category": "tool_execution", "detail": tool_errors[0], "downstream": "inspect_tool_result"}
    return {
        "category": "task_completion_or_prompt",
        "detail": "inspect conversation and trajectory against the task verifier",
        "downstream": "model_limit_schema_context_or_verifier_needs_review",
    }


def _model_turn_latencies(result: AgentResult) -> list[float | None]:
    durations: list[float | None] = []
    for event in result.trajectory.events:
        if event.kind != "model":
            continue
        if event.finished_at is None:
            durations.append(None)
        else:
            started = datetime.fromisoformat(event.started_at)
            finished = datetime.fromisoformat(event.finished_at)
            durations.append(round((finished - started).total_seconds(), 6))
    return durations


def run_suite(
    *,
    model_path: str,
    output_path: Path,
    max_steps: int = 12,
    max_seconds: float = 120.0,
    device: str = "auto",
) -> tuple[list[dict[str, Any]], int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = LocalTransformersClient(model_path, device=device, local_files_only=True)
    records: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as output:
        for task in MANUAL_SMOKE_TASKS:
            with TemporaryDirectory(prefix="terminal-slm-smoke-") as temporary:
                workspace = Path(temporary) / "workspace"
                shutil.copytree(REPOSITORY_ROOT, workspace, ignore=_ignore_copy_entries)
                toolbox = Toolbox(workspace)
                started = time.monotonic()
                result = AgentLoop(
                    client,
                    ToolExecutor(toolbox),
                    config=AgentConfig(
                        max_tool_calls=max_steps,
                        max_execution_time_seconds=max_seconds,
                    ),
                ).run(task.request)
                latency_seconds = time.monotonic() - started
                passed, verifier = task.verify(result, str(workspace))
                record = {
                    "schema_version": 1,
                    "run_kind": "manual_smoke_not_benchmark",
                    "task_id": task.task_id,
                    "task": task.request,
                    "model_path": model_path,
                    "device_requested": device,
                    "started_at": result.trajectory.started_at,
                    "latency_seconds": round(latency_seconds, 6),
                    "latency_scope": (
                        "AgentLoop wall time; includes lazy model load if needed, excludes repository copy and verification"
                    ),
                    "model_turn_latencies_seconds": _model_turn_latencies(result),
                    "stop_reason": result.stop_reason.value,
                    "final_outcome": "passed" if passed else "failed",
                    "verifier": verifier,
                    "diagnosis": _diagnose(result),
                    "metrics": _tool_metrics(result),
                    "conversation": list(result.messages),
                    "trajectory": result.trajectory.to_dict(),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                records.append(record)
                toolbox.close()
                print(
                    f"{task.task_id}: {record['final_outcome']} "
                    f"(stop={record['stop_reason']}, latency={record['latency_seconds']:.2f}s, "
                    f"tools={record['metrics']['valid_tool_calls']})"
                )

    failed = sum(record["final_outcome"] != "passed" for record in records)
    return records, failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the small manual Qwen3 harness smoke suite.")
    parser.add_argument("--model-path", default="Qwen/Qwen3-1.7B", help="local model path or cached identifier")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, help="JSONL trajectory path (default: timestamped research/results file)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_steps < 0:
        print("--max-steps must be zero or greater", file=sys.stderr)
        return 2
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        print("--max-seconds must be finite and greater than zero", file=sys.stderr)
        return 2
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or REPOSITORY_ROOT / "research" / "results" / f"manual_smoke_{timestamp}.jsonl"
    try:
        records, failed = run_suite(
            model_path=args.model_path,
            output_path=output_path,
            max_steps=args.max_steps,
            max_seconds=args.max_seconds,
            device=args.device,
        )
    except OSError as exc:
        print(f"could not prepare smoke suite: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"smoke suite stopped before completing: {exc}", file=sys.stderr)
        return 1

    print(f"Results: {output_path}")
    print(f"Passed: {len(records) - failed}/{len(records)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
