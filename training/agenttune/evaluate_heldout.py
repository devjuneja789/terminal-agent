"""Run the frozen Tiny Aya configuration on the fixed held-out split."""

from __future__ import annotations

import importlib.metadata
import base64
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.loop import AgentResult
from agent.state import AgentTrajectory, StopReason, TrajectoryEvent
from evals.benchmark.loader import BENCHMARK_ROOT, load_benchmark
from evals.benchmark.runner import run_task
from tools import Toolbox
from tools.definitions import get_tool_schemas, validate_tool_arguments
from training.agenttune.hf_gpu_smoke import (
    EVAL_CONFIG,
    MODEL_ID,
    MODEL_REVISION,
    TinyAyaPolicy,
    _canonical_sha256,
    verify_frozen_config,
    verify_runtime_versions,
)


def _event_json(log: Any) -> list[dict[str, Any]]:
    return [
        {
            "kind": str(getattr(event.kind, "value", event.kind)),
            "payload": event.payload,
        }
        for event in log
    ]


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hardware(torch: Any) -> dict[str, Any]:
    details: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(index)
            details["gpus"].append({
                "index": index,
                "name": gpu.name,
                "memory_bytes": gpu.total_memory,
                "compute_capability": f"{gpu.major}.{gpu.minor}",
            })
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            check=False,
        )
        details["nvidia_driver_versions"] = (
            [row.strip() for row in result.stdout.splitlines() if row.strip()]
            if result.returncode == 0 else None
        )
    except (OSError, subprocess.TimeoutExpired):
        details["nvidia_driver_versions"] = None
    details["packages"] = {
        name: _installed_version(name)
        for name in ("transformers", "accelerate", "agenttune", "huggingface_hub", "tokenizers")
    }
    return details


def _tool_result_success(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("success") is True
    return not (isinstance(value, str) and value.casefold().startswith("error:"))


def _bounded_result(value: dict[str, Any], limit: int) -> dict[str, Any]:
    output = dict(value)
    output["metadata"] = {}
    text_fields = [name for name in ("stdout", "stderr", "error") if isinstance(output.get(name), str)]
    for name in text_fields:
        output.setdefault(f"{name}_truncated", False)
    while len(json.dumps(output, ensure_ascii=False, separators=(",", ":"))) > limit:
        if not text_fields:
            break
        field = max(text_fields, key=lambda name: len(output[name]))
        current = output[field]
        fixed_size = len(json.dumps({**output, field: ""}, ensure_ascii=False, separators=(",", ":")))
        marker = "…[truncated]"
        keep = max(0, limit - fixed_size - len(marker) - 4)
        output[field] = current[:keep] + marker
        output[f"{field}_truncated"] = True
        if keep == 0:
            text_fields.remove(field)
    return output


def _run_one(task: Any, tokenizer: Any, model: Any, torch: Any, schemas: list[dict[str, Any]], *, run_id: str, metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    agent_started = time.monotonic()
    policy: TinyAyaPolicy | None = None
    episode: dict[str, Any] = {"agenttune_events": [], "tool_executions": [], "exception": None}

    def invoke(instruction: str, workspace: Path) -> AgentResult:
        from agenttune.agentic import DictToolHarness, Project, ReActStrategy

        toolbox = Toolbox(workspace)
        policy = TinyAyaPolicy(
            tokenizer,
            model,
            torch,
            schemas,
            time_limit_seconds=EVAL_CONFIG["agent"]["max_execution_seconds_per_task"],
        )
        episode["policy"] = policy
        dispatch: dict[str, Any] = {}

        for schema in schemas:
            tool_name = schema["function"]["name"]

            def make_tool(name: str):
                def execute(**arguments: Any) -> dict[str, Any]:
                    attempt: dict[str, Any] = {
                        "tool": name,
                        "arguments": arguments,
                        "schema_valid": False,
                    }
                    episode["tool_executions"].append(attempt)
                    try:
                        validated = validate_tool_arguments(name, arguments)
                        elapsed = time.monotonic() - episode.get("agent_started_monotonic", time.monotonic())
                        remaining = EVAL_CONFIG["agent"]["max_execution_seconds_per_task"] - elapsed
                        if remaining <= 0:
                            raise TimeoutError("per-task execution time limit reached")
                        if name in {"shell", "git"} and "timeout" in validated:
                            validated["timeout"] = min(validated["timeout"], remaining)
                        attempt["arguments"] = validated
                        attempt["schema_valid"] = True
                        result = _bounded_result(
                            getattr(toolbox, name)(**validated).as_dict(),
                            EVAL_CONFIG["agent"]["max_tool_output_chars"],
                        )
                    except (TypeError, ValueError) as exc:
                        result = {"success": False, "error": f"invalid tool arguments: {exc}"}
                    except TimeoutError as exc:
                        result = {"success": False, "error": str(exc)}
                    except Exception as exc:
                        result = {"success": False, "error": f"tool execution failed: {type(exc).__name__}: {exc}"}
                    attempt["result"] = result
                    return result
                return execute

            dispatch[tool_name] = make_tool(tool_name)

        harness = DictToolHarness(
            dispatch,
            max_steps=EVAL_CONFIG["agent"]["max_steps"],
            done_tool="finish",
        )
        strategy = ReActStrategy(policy, max_steps=EVAL_CONFIG["agent"]["max_steps"])
        project = Project(strategy=strategy, harness=harness)
        started_at = AgentTrajectory.timestamp()
        episode["agent_started_monotonic"] = time.monotonic()
        try:
            event_log = project.infer(instruction)
            events = _event_json(event_log)
            episode["agenttune_events"] = events
        except Exception as exc:
            episode["exception"] = f"{type(exc).__name__}: {exc}"
            events = []
        finally:
            episode["agent_elapsed_seconds"] = time.monotonic() - episode["agent_started_monotonic"]
            toolbox.close()

        native_events: list[TrajectoryEvent] = []
        final_response = ""
        action_index = 0
        for index, event in enumerate(events):
            kind = str(event["kind"]).lower()
            if "tool_call" not in kind:
                continue
            payload = event["payload"] if isinstance(event["payload"], dict) else {}
            action = payload.get("action", {})
            name = action.get("name")
            arguments = action.get("arguments", {})
            model_output = policy.raw_outputs[action_index] if action_index < len(policy.raw_outputs) else ""
            native_events.append(TrajectoryEvent(
                kind="model",
                started_at=AgentTrajectory.timestamp(),
                finished_at=AgentTrajectory.timestamp(),
                model_output=model_output,
            ))
            action_index += 1
            result_event = events[index + 1] if index + 1 < len(events) and "tool_result" in str(events[index + 1]["kind"]).lower() else None
            result_value = (
                result_event.get("payload", {}).get("output")
                if result_event and isinstance(result_event.get("payload"), dict)
                else None
            )
            if name == "finish":
                final_response = str(arguments.get("answer", ""))
                native_events.append(TrajectoryEvent(
                    kind="final",
                    started_at=AgentTrajectory.timestamp(),
                    finished_at=AgentTrajectory.timestamp(),
                    model_output=final_response,
                    executed=True,
                ))
            else:
                native_events.append(TrajectoryEvent(
                    kind="tool",
                    started_at=AgentTrajectory.timestamp(),
                    finished_at=AgentTrajectory.timestamp(),
                    tool_name=name,
                    tool_arguments=arguments if isinstance(arguments, dict) else {},
                    executed=True,
                    tool_result=result_value if isinstance(result_value, dict) else {"output": result_value},
                    error=None if _tool_result_success(result_value) else str(result_value),
                ))

        # If inference failed before Project returned an EventLog, retain the
        # actual model outputs and parse failures rather than emitting an empty trace.
        while action_index < len(policy.raw_outputs):
            native_events.append(TrajectoryEvent(
                kind="model",
                started_at=AgentTrajectory.timestamp(),
                finished_at=AgentTrajectory.timestamp(),
                model_output=policy.raw_outputs[action_index],
                error=policy.parse_errors[action_index],
            ))
            action_index += 1
        completed = bool(final_response) and episode["exception"] is None
        if completed:
            stop_reason = StopReason.COMPLETED
        elif policy.parse_errors and any(policy.parse_errors):
            stop_reason = StopReason.INVALID_TOOL_CALL
        elif episode["exception"]:
            stop_reason = (
                StopReason.EXECUTION_TIMEOUT
                if episode["exception"].startswith("TimeoutError:")
                else StopReason.MODEL_ERROR
            )
        else:
            stop_reason = StopReason.MAX_TOOL_CALLS
        trajectory = AgentTrajectory(
            user_request=instruction,
            started_at=started_at,
            events=native_events,
            final_response=final_response or None,
            finished_at=AgentTrajectory.timestamp(),
            stop_reason=stop_reason.value,
        )
        return AgentResult(
            final_text=final_response or None,
            messages=(),
            tool_calls=(),
            turns=len(policy.raw_outputs),
            stop_reason=stop_reason,
            trajectory=trajectory,
        )

    validation_timeout = EVAL_CONFIG["agent"]["validation_timeout_seconds"]
    with tempfile.TemporaryDirectory(prefix="terminal-slm-heldout-parent-") as parent:
        run = run_task(task, invoke, workspace_parent=Path(parent), validation_timeout=validation_timeout)

    policy = episode.get("policy")
    events = episode["agenttune_events"]
    selected_actions = policy.selected_actions if policy else []
    tool_actions = [action for action in selected_actions if action["name"] != "finish"]
    selected_tools = sorted({action["name"] for action in tool_actions})
    required_tools = set(task.success_condition.get("required_tools", []))
    unexpected_tools = set(selected_tools) - required_tools
    denominator = len(required_tools) + len(unexpected_tools)
    selection_accuracy = (
        len(required_tools & set(selected_tools)) / denominator if denominator else None
    )
    schema_valid = sum(bool(action["schema_valid"]) for action in tool_actions)
    malformed_outputs = sum(error is not None for error in (policy.parse_errors if policy else []))
    invalid_count = len(tool_actions) - schema_valid + malformed_outputs
    tool_attempt_count = len(tool_actions) + malformed_outputs
    prompt_tokens = policy.prompt_tokens_seen if policy else 0
    generated_tokens = policy.generated_tokens_seen if policy else 0
    model_outputs = policy.raw_outputs if policy else []
    model_prompts = policy.prompts if policy else []
    parse_errors = policy.parse_errors if policy else []
    response_adaptations = policy.response_adaptations if policy else []
    tool_executions = episode["tool_executions"]

    success = run.passed
    if episode["exception"]:
        failure_category = "invalid_tool_call" if policy and any(policy.parse_errors) else "agent_runtime_error"
    elif not success and any(not action["schema_valid"] for action in tool_actions):
        failure_category = "invalid_tool_call"
    elif not success and run.validation_timed_out:
        failure_category = "validation_timeout"
    elif not success and run.criterion_failures and any(item.startswith("validation_") for item in run.criterion_failures):
        failure_category = "validation_failure"
    elif not success and run.agent_stop_reason != StopReason.COMPLETED.value:
        failure_category = "agent_incomplete"
    elif not success:
        failed_execution = any(not _tool_result_success(row.get("result")) for row in tool_executions)
        failure_category = "tool_execution" if failed_execution else "task_contract_failure"
    else:
        failure_category = "none"

    elapsed = episode.get("agent_elapsed_seconds", time.monotonic() - agent_started)
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "benchmark_evaluation",
        "run_id": run_id,
        "benchmark_id": EVAL_CONFIG["evaluation"]["benchmark_id"],
        "config_id": EVAL_CONFIG["config_id"],
        "split": task.split,
        "task_id": task.task_id,
        "category": task.category,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "snapshot_sha256": task.snapshot_sha256,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": round(elapsed, 6),
        "success": success,
        "failure_category": failure_category,
        "criterion_failures": list(run.criterion_failures),
        "agent_stop_reason": run.agent_stop_reason,
        "agent_error": episode["exception"] or run.agent_error,
        "final_response": run.final_response,
        "success_condition": task.success_condition,
        "validation": {
            "command": list(task.validation_command),
            "exit_code": run.validation_exit_code,
            "timed_out": run.validation_timed_out,
            "stdout": run.validation_stdout,
            "stderr": run.validation_stderr,
            "seconds": round(run.validation_seconds, 6),
        },
        "tool_metrics": {
            "required_tools": sorted(required_tools),
            "selected_tools": selected_tools,
            "tool_selection_accuracy": selection_accuracy,
            "tool_calls_attempted": tool_attempt_count,
            "tool_calls_schema_valid": schema_valid,
            "tool_calls_invalid": invalid_count,
            "malformed_model_outputs": malformed_outputs,
            "argument_schema_validity": schema_valid / tool_attempt_count if tool_attempt_count else None,
            "invalid_tool_call_rate": invalid_count / tool_attempt_count if tool_attempt_count else 0.0,
            "tool_steps": sum(1 for event in events if "tool_call" in str(event["kind"]).lower() and event.get("payload", {}).get("action", {}).get("name") != "finish"),
            "model_turns": len(model_outputs),
        },
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": prompt_tokens + generated_tokens,
        },
        "trajectory": run.trajectory,
        "agenttune_trace": {
            "events": events,
            "model_outputs": model_outputs,
            "model_prompts": model_prompts,
            "parse_errors": parse_errors,
            "selected_actions": selected_actions,
            "ignored_trailing_model_text": policy.ignored_trailing_text if policy else [],
            "response_adaptations": response_adaptations,
            "tool_executions": tool_executions,
        },
        "experiment_metadata": metadata,
    }
    return record, {"validation_seconds": run.validation_seconds}


def _task_metric(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record["tool_metrics"]
    return {
        "tool_steps": metrics["tool_steps"],
        "model_turns": metrics["model_turns"],
        "tool_calls_attempted": metrics["tool_calls_attempted"],
        "tool_calls_schema_valid": metrics["tool_calls_schema_valid"],
        "tool_calls_invalid": metrics["tool_calls_invalid"],
        "tool_selection_accuracy": metrics["tool_selection_accuracy"],
        "argument_schema_validity": metrics["argument_schema_validity"],
        "invalid_tool_call_rate": metrics["invalid_tool_call_rate"],
        "total_tokens": record["token_usage"]["total_tokens"],
        "latency_seconds": record["latency_seconds"],
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [_task_metric(record) for record in records]
    total = len(rows)
    attempted = sum(row["tool_calls_attempted"] for row in rows)
    valid = sum(row["tool_calls_schema_valid"] for row in rows)
    invalid = sum(row["tool_calls_invalid"] for row in rows)
    required_total = sum(len(record["tool_metrics"]["required_tools"]) for record in records)
    required_selected = sum(
        len(set(record["tool_metrics"]["selected_tools"]) & set(record["tool_metrics"]["required_tools"]))
        for record in records
    )
    unexpected_total = sum(
        len(set(record["tool_metrics"]["selected_tools"]) - set(record["tool_metrics"]["required_tools"]))
        for record in records
    )
    selection_denominator = required_total + unexpected_total
    return {
        "task_count": total,
        "successful_tasks": sum(bool(record["success"]) for record in records),
        "task_success_rate": sum(bool(record["success"]) for record in records) / total if total else None,
        "tool_selection_accuracy": required_selected / selection_denominator if selection_denominator else None,
        "argument_schema_validity": valid / attempted if attempted else None,
        "invalid_tool_call_rate": invalid / attempted if attempted else 0.0,
        "average_tool_steps_per_task": sum(row["tool_steps"] for row in rows) / total if total else None,
        "average_model_turns_per_task": sum(row["model_turns"] for row in rows) / total if total else None,
        "average_tokens_per_task": sum(row["total_tokens"] for row in rows) / total if total else None,
        "average_latency_seconds": sum(row["latency_seconds"] for row in rows) / total if total else None,
        "failure_categories": dict(sorted(Counter(
            record["failure_category"] for record in records if not record["success"]
        ).items())),
    }


def _format_summary(summary: dict[str, Any], records: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    def number(value: Any) -> str:
        return "unavailable" if value is None else f"{value:.3f}"

    def percent(value: Any) -> str:
        return "unavailable" if value is None else f"{value:.1%}"

    lines = [
        "# Tiny Aya held-out baseline",
        "",
        "## Frozen configuration",
        "",
        f"- Config: `{metadata['config_id']}` (`{metadata['config_sha256']}`)",
        f"- Model: `{metadata['model']['repo_id']}` at `{metadata['model']['revision']}`",
        f"- Packages: `{json.dumps(metadata['packages'], sort_keys=True)}`",
        f"- GPU: `{json.dumps(metadata['hardware']['gpus'], sort_keys=True)}`",
        f"- Prompt SHA-256: `{metadata['prompt_sha256']}`",
        f"- Tool schema SHA-256: `{metadata['tool_schemas_sha256']}`",
        f"- Benchmark: `{metadata['benchmark_id']}` / `{metadata['split']}`",
        f"- Held-out IDs: {', '.join(metadata['task_ids'])}",
        "",
        "## Results",
        "",
        f"- Tasks: {summary['task_count']}",
        f"- Success rate: {percent(summary['task_success_rate'])}",
        f"- Tool selection accuracy: {percent(summary['tool_selection_accuracy'])}",
        f"- Argument/schema validity: {percent(summary['argument_schema_validity'])}",
        f"- Invalid tool-call rate: {percent(summary['invalid_tool_call_rate'])}",
        f"- Average tool steps/task: {number(summary['average_tool_steps_per_task'])}",
        f"- Average model turns/task: {number(summary['average_model_turns_per_task'])}",
        f"- Average tokens/task: {number(summary['average_tokens_per_task'])}",
        f"- Average latency/task: {number(summary['average_latency_seconds'])} seconds",
        "",
        "## Failure categories",
        "",
    ]
    lines.extend([f"- {key}: {value}" for key, value in summary["failure_categories"].items()] or ["- None"])
    lines.extend(["", "## Task outcomes", "", "| ID | Category | Outcome | Latency (s) | Failure |", "|---|---|---:|---:|---|"])
    lines.extend(
        f"| {row['task_id']} | {row['category']} | {'PASS' if row['success'] else 'FAIL'} | {row['latency_seconds']:.3f} | {row['failure_category']} |"
        for row in records
    )
    return "\n".join(lines) + "\n"


def _emit_task_record(record: dict[str, Any]) -> None:
    """Emit lossless, bounded log lines to survive HF console line limits."""
    encoded = base64.b64encode(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    ).decode("ascii")
    # HF Job logs truncate very long single lines. 12,000 base64 characters
    # stay comfortably below the observed 32 KiB cap after JSON framing.
    chunk_size = 12_000
    chunks = [encoded[index:index + chunk_size] for index in range(0, len(encoded), chunk_size)]
    for part, data in enumerate(chunks, 1):
        payload = {
            "task_id": record["task_id"],
            "part": part,
            "parts": len(chunks),
            "data": data,
        }
        print("BASELINE_TASK_RESULT_CHUNK=" + json.dumps(payload, separators=(",", ":")), flush=True)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("held-out baseline requires a Hugging Face GPU Job with CUDA")
    schemas = get_tool_schemas()
    verify_frozen_config(schemas)
    runtime_versions = verify_runtime_versions(torch)
    tasks = list(load_benchmark("held-out-eval"))
    expected_ids = EVAL_CONFIG["evaluation"]["task_ids"]
    if [task.task_id for task in tasks] != expected_ids:
        raise RuntimeError("loaded held-out IDs differ from the frozen evaluation configuration")
    manifest = json.loads((BENCHMARK_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest["benchmark_id"] != EVAL_CONFIG["evaluation"]["benchmark_id"]:
        raise RuntimeError("benchmark ID differs from the frozen evaluation configuration")

    model_load_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=EVAL_CONFIG["model"]["trust_remote_code"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=getattr(torch, EVAL_CONFIG["model"]["dtype"]),
        device_map=EVAL_CONFIG["model"]["device_map"],
        trust_remote_code=EVAL_CONFIG["model"]["trust_remote_code"],
    )
    model.eval()
    model_load_seconds = time.monotonic() - model_load_started

    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "config_id": EVAL_CONFIG["config_id"],
        "config_sha256": _canonical_sha256(EVAL_CONFIG),
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION, "dtype": EVAL_CONFIG["model"]["dtype"]},
        "packages": {
            "python": platform.python_version(),
            **runtime_versions,
            "transformers": _installed_version("transformers"),
            "accelerate": _installed_version("accelerate"),
            "agenttune": _installed_version("agenttune"),
            "huggingface_hub": _installed_version("huggingface_hub"),
        },
        "hardware": _hardware(torch),
        "model_load_seconds": round(model_load_seconds, 6),
        "benchmark_id": manifest["benchmark_id"],
        "split": "held-out-eval",
        "task_ids": expected_ids,
        "snapshot_hashes": sorted({task.snapshot_sha256 for task in tasks}),
        "prompt_sha256": EVAL_CONFIG["hashes"]["prompt_sha256"],
        "tool_schemas_sha256": EVAL_CONFIG["hashes"]["tool_schemas_sha256"],
        "evaluation_config": EVAL_CONFIG,
        "created_at": timestamp,
    }
    print("BASELINE_ENVIRONMENT=" + json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), flush=True)

    records: list[dict[str, Any]] = []
    for task in tasks:
        record, _ = _run_one(
            task,
            tokenizer,
            model,
            torch,
            schemas,
            run_id=run_id,
            metadata=metadata,
        )
        records.append(record)
        _emit_task_record(record)
        print(
            f"{task.task_id}: {'passed' if record['success'] else 'failed'} "
            f"({record['failure_category']}, {record['latency_seconds']:.2f}s)",
            file=sys.stderr,
            flush=True,
        )

    summary = _aggregate(records)
    print("BASELINE_SUMMARY=" + json.dumps({"run_id": run_id, "metrics": summary}, separators=(",", ":")), flush=True)
    print(_format_summary(summary, records, metadata), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
