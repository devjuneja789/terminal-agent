"""Stage a minimal HF Job input bundle without local weights or held-out data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COPY_PATHS = (
    "configs/tiny_aya_eval.json",
    "tools",
    "evals/__init__.py",
    "evals/benchmark/__init__.py",
    "evals/benchmark/loader.py",
    "evals/benchmark/train_support.jsonl",
    "evals/benchmark/snapshots/miniapp",
    "training/__init__.py",
    "training/agenttune/__init__.py",
    "training/agenttune/hf_gpu_smoke.py",
    "training/agenttune/run_smoke_job.sh",
)


def prepare(destination: Path) -> None:
    destination = destination.resolve()
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

    # The training loader validates split membership against this manifest.
    # Include support IDs only; leave the held-out task IDs out of the Job bundle.
    manifest = json.loads((ROOT / "evals/benchmark/manifest.json").read_text(encoding="utf-8"))
    support_only = {
        key: value for key, value in manifest.items() if key != "splits"
    }
    support_only["splits"] = {"train-support": manifest["splits"]["train-support"]}
    manifest_path = destination / "evals/benchmark/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(support_only, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.destination)
    print(f"Prepared support-only HF smoke bundle at {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
