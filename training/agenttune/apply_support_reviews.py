"""Rescore one support rollout and apply a human-review JSONL sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.benchmark.loader import load_benchmark
from training.agenttune.generate_support_rollouts import _rollout_summary
from training.agenttune.support_rewards import apply_human_review, load_review_sidecar, score_support_rollout


def apply_reviews(run_path: Path, review_path: Path) -> list[dict[str, Any]]:
    source_rows = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not source_rows:
        raise ValueError("rollout JSONL contains no records")
    run_ids = {row.get("run_id") for row in source_rows}
    if len(run_ids) != 1 or not isinstance(next(iter(run_ids)), str):
        raise ValueError("rollout JSONL must contain exactly one run ID")
    run_id = next(iter(run_ids))
    if any(row.get("split") != "train-support" for row in source_rows):
        raise ValueError("review application accepts train-support records only")
    tasks = {task.task_id: task for task in load_benchmark("train-support")}
    reviews = load_review_sidecar(review_path, run_id=run_id)
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        task_id = row["task_id"]
        if task_id not in tasks:
            raise ValueError(f"rollout contains non-support task {task_id}")
        if row.get("snapshot_sha256") != tasks[task_id].snapshot_sha256:
            raise ValueError(f"{task_id}: rollout snapshot differs from current support task")
        scored = score_support_rollout(tasks[task_id], row)
        review = reviews.get(task_id)
        if review is not None:
            scored = apply_human_review(
                scored,
                review,
                task_id=task_id,
                run_id=run_id,
                snapshot_sha256=row["snapshot_sha256"],
            )
        row["support_reward"] = scored
        row["success_at_generation"] = row.get("success")
        row["original_success_condition"] = row.get("success_condition")
        row["success_condition"] = tasks[task_id].success_condition
        row["success"] = scored["benchmark_contract_passed"]
        if row["success_at_generation"] and not row["success"]:
            row["failure_category"] = "contract_evidence_failure"
        row["criterion_failures"] = list(dict.fromkeys([
            *row.get("criterion_failures", []),
            *scored.get("contract_evidence_failures", []),
        ]))
        if isinstance(row.get("tool_metrics"), dict):
            required = set(tasks[task_id].success_condition.get("required_tools", []))
            selected = {
                item.get("name")
                for item in row.get("agenttune_trace", {}).get("selected_actions", [])
                if item.get("name") != "finish"
            }
            denominator = len(required) + len(selected - required)
            row["tool_metrics"]["required_tools"] = sorted(required)
            row["tool_metrics"]["tool_selection_accuracy"] = (
                len(required & selected) / denominator if denominator else None
            )
        row.setdefault("support_audit", {})["reward"] = scored
        row["support_audit"]["human_review"] = scored["human_review"]
        row["support_audit"]["training_eligible"] = False
        row["training_target_eligible"] = False
        if row.get("failure_category") == "agent_loop_guard" or row.get("agent_stop_reason") == "agent_loop_guard":
            # TinyAyaPolicy inserts a harness-authored finish response when it
            # blocks a repeated action. Preserve the raw trace, but make this
            # record ineligible as a training target rather than teaching the
            # model to emit the harness fallback.
            row["training_target_excluded"] = True
            row["training_target_exclusion_reason"] = "harness_generated_loop_guard_response"
        rows.append(row)
    unknown_reviews = set(reviews) - {row["task_id"] for row in source_rows}
    if unknown_reviews:
        raise ValueError(f"review sidecar contains tasks absent from run: {sorted(unknown_reviews)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = apply_reviews(args.run, args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "run_id": rows[0]["run_id"],
        "source": str(args.run),
        "review": str(args.review),
        "reviewed_output": str(args.output),
        "metrics": _rollout_summary(rows),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
