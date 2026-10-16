"""Load the fixed Terminal-SLM benchmark without crossing its data split."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


BENCHMARK_ROOT = Path(__file__).resolve().parent
SPLITS: dict[str, str] = {
    "train-support": "train_support.jsonl",
    "held-out-eval": "heldout_eval.jsonl",
}


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    category: str
    split: Literal["train-support", "held-out-eval"]
    snapshot: str
    snapshot_sha256: str
    instruction: str
    expected_outcome: str
    validation_command: tuple[str, ...]
    success_condition: dict[str, Any]
    reference_patch: str | None = None

    @property
    def snapshot_path(self) -> Path:
        path = (BENCHMARK_ROOT / self.snapshot).resolve()
        if not path.is_relative_to(BENCHMARK_ROOT.resolve()):
            raise ValueError(f"snapshot escapes benchmark root: {self.snapshot}")
        return path


def _read_manifest() -> dict[str, Any]:
    return json.loads((BENCHMARK_ROOT / "manifest.json").read_text(encoding="utf-8"))


def snapshot_digest(root: Path) -> str:
    digest = hashlib.sha256()
    entries = list(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError(f"repository snapshot must not contain symlinks: {root}")
    for path in sorted(item for item in entries if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def load_benchmark(split: str) -> tuple[BenchmarkTask, ...]:
    """Return one split only; held-out tasks cannot be requested as training data."""
    if split not in SPLITS:
        raise ValueError(f"unknown benchmark split {split!r}; choose one of {tuple(SPLITS)}")
    manifest = _read_manifest()
    filename = SPLITS[split]
    path = BENCHMARK_ROOT / filename
    tasks: list[BenchmarkTask] = []
    seen: set[str] = set()
    verified_snapshots: dict[Path, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != split:
            raise ValueError(f"{filename}:{line_number}: task has wrong split")
        task = BenchmarkTask(
            task_id=row["task_id"],
            category=row["category"],
            split=row["split"],
            snapshot=row["snapshot"],
            snapshot_sha256=row["snapshot_sha256"],
            instruction=row["instruction"],
            expected_outcome=row["expected_outcome"],
            validation_command=tuple(row["validation_command"]),
            success_condition=row["success_condition"],
            reference_patch=row.get("reference_patch"),
        )
        if not task.task_id or task.task_id in seen:
            raise ValueError(f"{filename}:{line_number}: empty or duplicate task ID")
        if not task.validation_command:
            raise ValueError(f"{task.task_id}: validation_command must not be empty")
        if not task.snapshot_path.is_dir():
            raise FileNotFoundError(f"{task.task_id}: snapshot directory does not exist")
        snapshot_path = task.snapshot_path
        actual = verified_snapshots.get(snapshot_path)
        if actual is None:
            actual = snapshot_digest(snapshot_path)
            verified_snapshots[snapshot_path] = actual
        if actual != task.snapshot_sha256:
            raise ValueError(f"{task.task_id}: repository snapshot digest mismatch")
        seen.add(task.task_id)
        tasks.append(task)

    expected_ids = set(manifest["splits"][split])
    if seen != expected_ids:
        raise ValueError(f"{split}: task IDs differ from permanent manifest")
    return tuple(tasks)


def training_task_ids() -> tuple[str, ...]:
    """Expose only train-support IDs to training-data preparation code."""
    return tuple(task.task_id for task in load_benchmark("train-support"))
