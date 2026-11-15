"""Build a support-only candidate-rollout bundle with no held-out task data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PILOT_TASK_IDS = (
    "B001", "B006", "B011", "B016", "B021",
    "B026", "B031", "B036", "B041", "B046",
)
COPY_PATHS = (
    "agent/__init__.py",
    "agent/loop.py",
    "agent/parser.py",
    "agent/prompts.py",
    "agent/state.py",
    "evals/__init__.py",
    "evals/benchmark/__init__.py",
    "evals/benchmark/train_support.jsonl",
    "evals/benchmark/loader.py",
    "evals/benchmark/manifest.json",
    "evals/benchmark/runner.py",
    "evals/benchmark/snapshots/miniapp",
    "model/__init__.py",
    "model/client.py",
    "tools",
    "training/__init__.py",
    "training/agenttune/__init__.py",
    "training/agenttune/evaluate_heldout.py",
    "training/agenttune/generate_support_rollouts.py",
    "training/agenttune/hf_gpu_smoke.py",
    "training/agenttune/run_support_job.sh",
    "training/agenttune/support_prompt.py",
    "training/agenttune/support_rewards.py",
)


def _sha256(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def prepare(destination: Path, task_ids: tuple[str, ...] = PILOT_TASK_IDS) -> tuple[str, ...]:
    from training.agenttune.support_rewards import PENALTIES
    from training.agenttune.support_prompt import SUPPORT_SYSTEM_PROMPT

    destination = destination.resolve()
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("support pilot task IDs must be a non-empty unique list")
    if not set(task_ids).issubset(PILOT_TASK_IDS):
        raise ValueError("custom support pilot IDs must be selected from the predeclared support pilot")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"bundle destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    for relative in COPY_PATHS:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        else:
            shutil.copy2(source, target)

    source_config = json.loads((ROOT / "configs/tiny_aya_eval.json").read_text(encoding="utf-8"))
    config = {
        key: source_config[key]
        for key in ("model", "runtime", "generation", "agent", "prompt", "hashes")
    }
    config["prompt"] = {"system": SUPPORT_SYSTEM_PROMPT}
    config["hashes"]["prompt_sha256"] = _sha256(SUPPORT_SYSTEM_PROMPT)
    config.update({
        "config_id": "terminal-slm-tiny-aya-support-rollout-v6",
        "prompt_variant": "support-v5-repeat-finish-retry",
        "support_task_revision": "2026-11-12-gold-contracts-r6",
        "source_eval_config_sha256": _sha256(source_config),
        "support_taskset_sha256": hashlib.sha256(
            (ROOT / "evals/benchmark/train_support.jsonl").read_bytes()
        ).hexdigest(),
        "evaluation": {
            "benchmark_id": source_config["evaluation"]["benchmark_id"],
            "split": "train-support",
            "task_ids": list(task_ids),
        },
        "rollout": {
            "mode": "candidate_generation_only",
            "selection_policy": (
                "one preselected support task from each benchmark category"
                if task_ids == PILOT_TASK_IDS
                else "explicit subset of preselected train-support pilot tasks"
            ),
            "task_ids": list(task_ids),
            "human_review_required": True,
            "training_eligible_without_review": False,
        },
        "reward": {
            "change_required_categories": [
                "simple_bug_fix", "failing_test_fix", "validation_change",
                "api_route_fix", "configuration_update",
            ],
            "factual_review_categories": [
                "repository_navigation", "function_discovery", "configuration_discovery",
                "environment_variable_discovery", "explaining_test_failure",
            ],
            "weights": PENALTIES,
            "primary_success_evidence": "task contract AND isolated validation; change tasks additionally require a pristine validator failure and changed task files",
        },
        "rollout_controls": {
            "one_action_per_model_turn": True,
            "require_tool_observation_before_next_action": True,
            "preserve_first_valid_tool_action_when_trailing_prose_exists": True,
            "stop_repeated_identical_actions_without_intervening_edit": True,
            "on_repeat": "ask_model_once_to_finish_from_observations; reject_tools; empty_harness_stop_if_retry_invalid",
            "normalize_python_dict_observations_to_json": True,
        },
        "job": {"flavor": "t4-small", "timeout": "40m"},
    })
    config_dir = destination / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "tiny_aya_support_rollout.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = json.loads((ROOT / "evals/benchmark/manifest.json").read_text(encoding="utf-8"))
    support_ids = tuple(manifest["splits"]["train-support"])
    if not set(PILOT_TASK_IDS).issubset(support_ids):
        raise ValueError("pilot includes an ID outside train-support")
    support_only = {key: value for key, value in manifest.items() if key != "splits"}
    support_only["splits"] = {"train-support": list(support_ids)}
    manifest_path = destination / "evals/benchmark/manifest.json"
    manifest_path.write_text(json.dumps(support_only, indent=2) + "\n", encoding="utf-8")

    tasks = [
        json.loads(line)
        for line in (destination / "evals/benchmark/train_support.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = tuple(task["task_id"] for task in tasks)
    if set(ids) != set(support_ids) or any(task["split"] != "train-support" for task in tasks):
        raise ValueError("support task file and support-only manifest disagree")
    selected = [task for task in tasks if task["task_id"] in task_ids]
    if len(selected) != len(task_ids) or len({task["category"] for task in selected}) != len(selected):
        raise ValueError("pilot IDs must resolve to unique train-support categories")

    forbidden = (
        destination / "evals/benchmark/heldout_eval.jsonl",
        destination / "configs/tiny_aya_eval.json",
    )
    if any(path.exists() for path in forbidden):
        raise ValueError("support bundle unexpectedly contains held-out task/config data")
    return task_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="predeclared train-support pilot task ID (repeatable); defaults to the full ten-task pilot",
    )
    args = parser.parse_args()
    task_ids = prepare(args.destination, tuple(args.task_ids) if args.task_ids else PILOT_TASK_IDS)
    print(f"Prepared support-only rollout bundle for {len(task_ids)} tasks: {', '.join(task_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
