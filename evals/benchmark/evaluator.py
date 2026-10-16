"""Run selected benchmark tasks and persist their real trajectories/results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from agent.loop import AgentLoop
from agent.prompts import DEFAULT_SYSTEM_PROMPT
from agent.state import AgentConfig
from evals.benchmark.loader import BENCHMARK_ROOT, SPLITS, BenchmarkTask, load_benchmark
from evals.benchmark.metrics import aggregate_metrics, task_metrics
from evals.benchmark.runner import BenchmarkRun, run_task
from model.local import LocalTransformersClient
from tools import Toolbox
from tools.definitions import get_tool_schemas
from tools.executor import ToolExecutor


def _failure_category(run: BenchmarkRun, trajectory: dict[str, Any] | None) -> str:
    if run.passed:
        return "none"
    if run.agent_error:
        return "agent_runtime_error"
    if run.agent_stop_reason == "invalid_tool_call":
        return "invalid_tool_call"
    if run.agent_stop_reason == "model_error":
        return "model_inference"
    if run.agent_stop_reason == "execution_timeout":
        return "agent_timeout"
    if trajectory:
        failed_tools = [
            event for event in trajectory.get("events", [])
            if event.get("kind") == "tool"
            and event.get("executed")
            and not (event.get("tool_result") or {}).get("success")
        ]
        if failed_tools:
            return "tool_execution"
    if run.validation_timed_out:
        return "validation_timeout"
    if "validation_command_failed" in run.criterion_failures:
        return "validation_failure"
    if "agent_incomplete" in run.criterion_failures:
        return "agent_incomplete"
    return "task_contract_failure"


def _token_snapshot(client: Any) -> tuple[int, int] | None:
    if not hasattr(client, "prompt_tokens_seen") or not hasattr(client, "generated_tokens_seen"):
        return None
    return int(client.prompt_tokens_seen), int(client.generated_tokens_seen)


def _token_delta(client: Any, before: tuple[int, int] | None) -> dict[str, int] | None:
    after = _token_snapshot(client)
    if before is None or after is None:
        return None
    prompt, generated = after[0] - before[0], after[1] - before[1]
    return {
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "total_tokens": prompt + generated,
    }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_cuda(torch_module: Any | None = None) -> Any:
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("--device cuda requires PyTorch with CUDA support") from exc
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() < 1:
        raise RuntimeError("CUDA was requested but this session cannot access a CUDA device")
    return torch_module


def _hardware_metadata(torch_module: Any) -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": None,
        "torch_version": getattr(torch_module, "__version__", None),
        "cuda_runtime_version": getattr(torch_module.version, "cuda", None),
        "cuda_available": bool(torch_module.cuda.is_available()),
        "gpus": [],
    }
    try:
        hardware["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    if hardware["cuda_available"]:
        for index in range(torch_module.cuda.device_count()):
            props = torch_module.cuda.get_device_properties(index)
            hardware["gpus"].append({
                "index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
            })
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            check=False,
        )
        if driver.returncode == 0:
            hardware["nvidia_driver_versions"] = [line.strip() for line in driver.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired):
        hardware["nvidia_driver_versions"] = None
    return hardware


def _model_metadata(model_path: str, client: LocalTransformersClient) -> dict[str, Any]:
    requested_path = Path(model_path)
    resolved = requested_path.resolve() if requested_path.exists() else None
    config: dict[str, Any] = {}
    files: list[dict[str, Any]] = []
    if resolved is not None and resolved.is_dir():
        config_path = resolved / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            files.append({"path": "config.json", "size_bytes": config_path.stat().st_size})
        for path in sorted(resolved.glob("*.safetensors")):
            files.append({"path": path.name, "size_bytes": path.stat().st_size})
        index_path = resolved / "model.safetensors.index.json"
        if index_path.is_file():
            files.append({"path": index_path.name, "size_bytes": index_path.stat().st_size})
    packages: dict[str, str | None] = {}
    for package in ("torch", "transformers", "tokenizers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "requested_id_or_path": model_path,
        "resolved_path": str(resolved) if resolved is not None else None,
        "config_name_or_path": config.get("_name_or_path"),
        "config_commit_hash": config.get("_commit_hash"),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "config_sha256": hashlib.sha256((resolved / "config.json").read_bytes()).hexdigest()
        if resolved is not None and (resolved / "config.json").is_file() else None,
        "weight_file_inventory": files,
        "packages": packages,
        "inference_configuration": {
            "device_requested": client.device,
            "dtype": client.dtype,
            "quantization": "none",
            "max_new_tokens": client.max_new_tokens,
            "enable_thinking": client.enable_thinking,
            "local_files_only": client.local_files_only,
            "do_sample": False,
            "use_cache": True,
        },
    }


def _benchmark_metadata(split: str) -> dict[str, Any]:
    manifest = json.loads((BENCHMARK_ROOT / "manifest.json").read_text(encoding="utf-8"))
    return {
        "benchmark_id": manifest["benchmark_id"],
        "split": split,
        "split_task_ids": manifest["splits"][split],
        "snapshot_hashes": sorted({
            task.snapshot_sha256 for task in load_benchmark(split)
        }),
    }


def _experiment_metadata(
    *, split: str, model_path: str, device: str, client: LocalTransformersClient,
    max_steps: int, max_seconds: float, validation_timeout: float,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    schemas = get_tool_schemas()
    return {
        "model": _model_metadata(model_path, client),
        "benchmark": _benchmark_metadata(split),
        "prompt": {
            "name": "DEFAULT_SYSTEM_PROMPT",
            "text": DEFAULT_SYSTEM_PROMPT,
            "sha256": hashlib.sha256(DEFAULT_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "enable_thinking": client.enable_thinking,
        },
        "tool_schemas": {"definitions": schemas, "sha256": _sha256_json(schemas)},
        "agent_configuration": {
            "max_tool_calls": max_steps,
            "max_execution_time_seconds": max_seconds,
            "validation_timeout_seconds": validation_timeout,
        },
        "hardware": hardware,
    }


def _one_task_record(
    task: BenchmarkTask,
    client: LocalTransformersClient,
    *,
    run_id: str,
    max_steps: int,
    max_seconds: float,
    validation_timeout: float,
    workspace_parent: Path,
    model_path: str,
    device: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    token_start = _token_snapshot(client)

    def invoke_agent(instruction: str, workspace: Path):
        toolbox = Toolbox(workspace)
        try:
            loop = AgentLoop(
                client,
                ToolExecutor(toolbox),
                config=AgentConfig(
                    max_tool_calls=max_steps,
                    max_execution_time_seconds=max_seconds,
                ),
            )
            return loop.run(instruction)
        finally:
            toolbox.close()

    started_at = datetime.now(timezone.utc).isoformat()
    run = run_task(
        task,
        invoke_agent,
        workspace_parent=workspace_parent,
        validation_timeout=validation_timeout,
    )
    trajectory = run.trajectory
    success_condition = task.success_condition
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "benchmark_evaluation",
        "run_id": run_id,
        "benchmark_id": metadata["benchmark"]["benchmark_id"],
        "split": task.split,
        "task_id": task.task_id,
        "category": task.category,
        "model_path": model_path,
        "device_requested": device,
        "experiment_metadata": metadata,
        "snapshot_sha256": task.snapshot_sha256,
        "started_at": started_at,
        "latency_seconds": round(run.elapsed_seconds, 6),
        "validation_seconds": round(run.validation_seconds, 6),
        "success": run.passed,
        "failure_category": _failure_category(run, trajectory),
        "criterion_failures": list(run.criterion_failures),
        "agent_stop_reason": run.agent_stop_reason,
        "agent_error": run.agent_error,
        "final_response": run.final_response,
        "success_condition": success_condition,
        "validation": {
            "command": list(task.validation_command),
            "exit_code": run.validation_exit_code,
            "timed_out": run.validation_timed_out,
            "stdout": run.validation_stdout,
            "stderr": run.validation_stderr,
        },
        "token_usage": _token_delta(client, token_start),
        "trajectory": trajectory,
    }
    record["metrics"] = task_metrics(record)
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    return record


def _format_summary(
    summary: dict[str, Any], *, split: str, records: list[dict[str, Any]], metadata: dict[str, Any]
) -> str:
    model = metadata["model"]
    hardware = metadata["hardware"]
    prompt = metadata["prompt"]
    schemas = metadata["tool_schemas"]
    lines = [
        f"# Terminal-SLM benchmark run: {split}",
        "",
        "This summary is calculated from the JSONL task records in this run.",
        "",
        "## Configuration",
        "",
        f"- Benchmark: {metadata['benchmark']['benchmark_id']}",
        f"- Model path: {model['requested_id_or_path']}",
        f"- Model type / commit: {model['model_type']} / {model['config_commit_hash']}",
        f"- Model config SHA-256: {model['config_sha256']}",
        f"- Inference: {model['inference_configuration']['dtype']}, quantization {model['inference_configuration']['quantization']}, device {model['inference_configuration']['device_requested']}",
        f"- Packages: {json.dumps(model['packages'], sort_keys=True)}",
        f"- Prompt: {prompt['name']} SHA-256 `{prompt['sha256']}`",
        f"- Tool schemas SHA-256: `{schemas['sha256']}`",
        f"- Hardware: {json.dumps(hardware, sort_keys=True)}",
        f"- Split task IDs: {', '.join(metadata['benchmark']['split_task_ids'])}",
        "",
        f"- Tasks: {summary['task_count']}",
        f"- Successes: {summary['successful_tasks']}",
        f"- Success rate: {_percent(summary['task_success_rate'])}",
        f"- Tool selection accuracy: {_percent(summary['tool_selection_accuracy'])}",
        f"- Argument/schema validity: {_percent(summary['argument_schema_validity'])}",
        f"- Invalid tool-call rate: {_percent(summary['invalid_tool_call_rate'])}",
        f"- Average tool steps/task: {_number(summary['average_tool_steps_per_task'])}",
        f"- Average model turns/task: {_number(summary['average_model_turns_per_task'])}",
        f"- Average tokens/task: {_number(summary['average_tokens_per_task'])} ({summary['token_metric_tasks']} tasks measured)",
        f"- Average latency/task: {_number(summary['average_latency_seconds'])} seconds",
        "",
        "## Failure categories",
        "",
    ]
    if summary["failure_categories"]:
        lines.extend(f"- {category}: {count}" for category, count in summary["failure_categories"].items())
    else:
        lines.append("- None")
    lines.extend(["", "## Tasks", "", "| Task ID | Category | Outcome | Latency (s) | Failure |", "|---|---|---:|---:|---|"])
    lines.extend(
        f"| {row['task_id']} | {row['category']} | {'PASS' if row['success'] else 'FAIL'} | "
        f"{row['latency_seconds']:.3f} | {row['failure_category']} |"
        for row in records
    )
    return "\n".join(lines) + "\n"


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def _percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.1%}"


def evaluate(
    *,
    split: str,
    model_path: str,
    output_path: Path,
    summary_path: Path,
    task_ids: Sequence[str] | None = None,
    limit: int | None = None,
    max_steps: int = 16,
    max_seconds: float = 600.0,
    validation_timeout: float = 120.0,
    device: str = "auto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if max_steps < 0 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_steps must be nonnegative and max_seconds finite and positive")

    tasks = list(load_benchmark(split))
    if task_ids:
        by_id = {task.task_id: task for task in tasks}
        unknown = set(task_ids) - by_id.keys()
        if unknown:
            raise ValueError(f"task IDs are not in split {split}: {', '.join(sorted(unknown))}")
        tasks = [by_id[task_id] for task_id in task_ids]
    elif limit is not None:
        tasks = tasks[:limit]
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("no benchmark tasks selected")

    if device == "cuda":
        torch_module = _require_cuda()
    else:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("benchmark evaluation requires the configured PyTorch runtime") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    client = LocalTransformersClient(model_path, device=device, local_files_only=True)
    metadata = _experiment_metadata(
        split=split,
        model_path=model_path,
        device=device,
        client=client,
        max_steps=max_steps,
        max_seconds=max_seconds,
        validation_timeout=validation_timeout,
        hardware=_hardware_metadata(torch_module),
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as output:
        for task in tasks:
            with tempfile.TemporaryDirectory(prefix="terminal-slm-benchmark-") as parent:
                record = _one_task_record(
                    task,
                    client,
                    run_id=run_id,
                    max_steps=max_steps,
                    max_seconds=max_seconds,
                    validation_timeout=validation_timeout,
                    workspace_parent=Path(parent),
                    model_path=model_path,
                    device=device,
                    metadata=metadata,
                )
            # The isolated workspace is gone before a result record is emitted.
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            records.append(record)
            print(
                f"{task.task_id}: {'passed' if record['success'] else 'failed'} "
                f"({record['failure_category']}, {record['latency_seconds']:.2f}s)",
                file=sys.stderr,
            )

    summary = aggregate_metrics(records)
    summary_document = {
        "schema_version": 1,
        "run_id": run_id,
        "run_kind": "benchmark_evaluation_summary",
        "benchmark_id": metadata["benchmark"]["benchmark_id"],
        "split": split,
        "model_path": model_path,
        "task_ids": [row["task_id"] for row in records],
        "experiment_metadata": metadata,
        "metrics": summary,
    }
    summary_path.write_text(
        _format_summary(summary, split=split, records=records, metadata=metadata), encoding="utf-8"
    )
    summary_path.with_suffix(".json").write_text(
        json.dumps(summary_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return records, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Terminal-SLM on one fixed benchmark split.")
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--task-id", action="append", help="select a task in this split; may be repeated")
    parser.add_argument("--limit", type=int, help="run only the first N selected tasks")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--validation-timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, help="JSONL output path")
    parser.add_argument("--summary", type=Path, help="Markdown summary output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(__file__).resolve().parent / "results" / args.split / f"run_{timestamp}"
    output_path = args.output or base.with_suffix(".jsonl")
    summary_path = args.summary or base.with_suffix(".md")
    try:
        records, summary = evaluate(
            split=args.split,
            model_path=args.model_path,
            output_path=output_path,
            summary_path=summary_path,
            task_ids=args.task_id,
            limit=args.limit,
            max_steps=args.max_steps,
            max_seconds=args.max_seconds,
            validation_timeout=args.validation_timeout,
            device=args.device,
        )
    except Exception as exc:
        print(f"benchmark evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Results: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Tasks: {len(records)}, success rate: {_percent(summary['task_success_rate'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
