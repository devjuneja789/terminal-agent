"""Stage a held-out-only baseline bundle for a disposable Hugging Face Job."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COPY_PATHS = (
    "agent/__init__.py",
    "agent/loop.py",
    "agent/parser.py",
    "agent/prompts.py",
    "agent/state.py",
    "configs/tiny_aya_eval.json",
    "evals/__init__.py",
    "evals/benchmark/__init__.py",
    "evals/benchmark/heldout_eval.jsonl",
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
    "training/agenttune/hf_gpu_smoke.py",
    "training/agenttune/run_baseline_job.sh",
)


def prepare(destination: Path) -> tuple[str, ...]:
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"bundle destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for relative in COPY_PATHS:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, target)

    manifest = json.loads((ROOT / "evals/benchmark/manifest.json").read_text(encoding="utf-8"))
    heldout_ids = tuple(manifest["splits"]["held-out-eval"])
    heldout_only = {key: value for key, value in manifest.items() if key != "splits"}
    heldout_only["splits"] = {"held-out-eval": list(heldout_ids)}
    (destination / "evals/benchmark/manifest.json").write_text(
        json.dumps(heldout_only, indent=2) + "\n", encoding="utf-8"
    )
    task_ids = tuple(
        json.loads(line)["task_id"]
        for line in (destination / "evals/benchmark/heldout_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if task_ids != heldout_ids:
        raise ValueError("held-out task file differs from frozen manifest order")
    if (destination / "evals/benchmark/train_support.jsonl").exists():
        raise ValueError("support task data must not be present in the held-out bundle")
    return heldout_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    task_ids = prepare(args.destination)
    print(f"Prepared held-out-only bundle with {len(task_ids)} tasks: {', '.join(task_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
