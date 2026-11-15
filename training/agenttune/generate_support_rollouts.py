"""Generate reviewed-candidate trajectories from a fixed support-only pilot."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.benchmark.loader import BENCHMARK_ROOT, load_benchmark
from tools.definitions import get_tool_schemas
from training.agenttune.evaluate_heldout import _emit_task_record, _hardware, _run_one
from training.agenttune.hf_gpu_smoke import (
    EVAL_CONFIG,
    MODEL_ID,
    MODEL_REVISION,
    _canonical_sha256,
    verify_frozen_config,
    verify_runtime_versions,
)
from training.agenttune.support_rewards import score_support_rollout


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _rollout_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row["support_reward"]["reward"] for row in records if row["support_reward"]["reward"] is not None]
    failures: Counter[str] = Counter()
    for row in records:
        reward = row["support_reward"]
        if reward["benchmark_contract_passed"]:
            continue
        category = row.get("failure_category")
        if category in {None, "none"}:
            category = "contract_evidence_failure" if reward.get("contract_evidence_failures") else "task_contract_failure"
        failures[str(category)] += 1
    return {
        "candidate_count": len(records),
        "benchmark_contract_passes": sum(bool(row["support_reward"]["benchmark_contract_passed"]) for row in records),
        "automated_task_successes": sum(bool(row["support_reward"]["automated_task_success"]) for row in records),
        "scored_candidates": len(scored),
        "human_review_required": sum(bool(row["support_reward"]["human_review"]["status"] == "pending") for row in records),
        "training_eligible_candidates": sum(bool(row["support_reward"]["human_review"]["training_eligible"]) for row in records),
        "average_nonnull_reward": sum(scored) / len(scored) if scored else None,
        "reward_status_counts": dict(sorted(Counter(row["support_reward"]["reward_status"] for row in records).items())),
        "failure_categories": dict(sorted(failures.items())),
    }


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("support rollout pilot requires a Hugging Face CUDA Job")
    schemas = get_tool_schemas()
    verify_frozen_config(schemas)
    runtime_versions = verify_runtime_versions(torch)

    manifest = json.loads((BENCHMARK_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["splits"]) != {"train-support"}:
        raise RuntimeError("support rollout bundle manifest must expose train-support only")
    if (BENCHMARK_ROOT / "heldout_eval.jsonl").exists():
        raise RuntimeError("support rollout bundle must not contain held-out task records")

    support_tasks = list(load_benchmark("train-support"))
    support_ids = {task.task_id for task in support_tasks}
    selected_ids = EVAL_CONFIG["rollout"]["task_ids"]
    if len(selected_ids) != len(set(selected_ids)) or not set(selected_ids).issubset(support_ids):
        raise RuntimeError("support rollout selection includes a duplicate or non-support task ID")
    tasks_by_id = {task.task_id: task for task in support_tasks}
    tasks = [tasks_by_id[task_id] for task_id in selected_ids]
    if len({task.category for task in tasks}) != len(tasks):
        raise RuntimeError("support rollout pilot is not category-balanced")

    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = time.monotonic()
    model_started = time.monotonic()
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
    model_load_seconds = time.monotonic() - model_started

    metadata = {
        "run_kind": "support_rollout_candidate_generation",
        "config_id": EVAL_CONFIG["config_id"],
        "config_sha256": _canonical_sha256(EVAL_CONFIG),
        "source_eval_config_sha256": EVAL_CONFIG["source_eval_config_sha256"],
        "support_task_revision": EVAL_CONFIG["support_task_revision"],
        "support_taskset_sha256": EVAL_CONFIG["support_taskset_sha256"],
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION, "dtype": EVAL_CONFIG["model"]["dtype"]},
        "packages": {
            "python": platform.python_version(),
            **runtime_versions,
            "huggingface_hub": _package_version("huggingface_hub"),
        },
        "hardware": _hardware(torch),
        "model_load_seconds": round(model_load_seconds, 6),
        "benchmark_id": manifest["benchmark_id"],
        "split": "train-support",
        "task_ids": selected_ids,
        "support_task_count": len(support_tasks),
        "selection_policy": EVAL_CONFIG["rollout"]["selection_policy"],
        "snapshot_hashes": sorted({task.snapshot_sha256 for task in tasks}),
        "prompt_sha256": EVAL_CONFIG["hashes"]["prompt_sha256"],
        "tool_schemas_sha256": EVAL_CONFIG["hashes"]["tool_schemas_sha256"],
        "evaluation_config": EVAL_CONFIG,
        "created_at": timestamp,
    }
    print("SUPPORT_ROLLOUT_ENVIRONMENT=" + json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), flush=True)

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
            capture_support_audit=True,
        )
        record["run_kind"] = "support_rollout_candidate"
        record["support_reward"] = score_support_rollout(task, record)
        record["support_audit"]["reward"] = record["support_reward"]
        record["support_audit"]["human_review"] = record["support_reward"]["human_review"]
        record["support_audit"]["training_eligible"] = False
        record["training_target_eligible"] = False
        if record.get("final_response_source") == "harness_empty_stop":
            record["training_target_excluded"] = True
            record["training_target_exclusion_reason"] = "no_model_final_response_after_repeat_recovery"
        if record["task_id"] not in support_ids or record["split"] != "train-support":
            raise RuntimeError(f"refusing to emit non-support rollout record {record['task_id']}")
        if record["support_reward"]["reward"] is not None and not record["support_reward"]["automated_task_success"] and record["support_reward"]["reward"] > 0:
            raise RuntimeError("reward integrity check failed: failed task received positive reward")
        records.append(record)
        _emit_task_record(record, marker="SUPPORT_ROLLOUT_TASK_CHUNK")
        print(
            f"{task.task_id}: contract={'pass' if record['support_reward']['benchmark_contract_passed'] else 'fail'} "
            f"reward={record['support_reward']['reward']} ({record['support_reward']['reward_status']})",
            file=sys.stderr,
            flush=True,
        )

    if [record["task_id"] for record in records] != selected_ids:
        raise RuntimeError("emitted rollout order differs from the frozen support pilot config")
    summary = {
        "run_id": run_id,
        "config_id": EVAL_CONFIG["config_id"],
        "split": "train-support",
        "task_ids": selected_ids,
        "metrics": _rollout_summary(records),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "heldout_accessed": False,
    }
    print("SUPPORT_ROLLOUT_SUMMARY=" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
