from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.benchmark import load_benchmark
from evals.benchmark.loader import training_task_ids
from evals.benchmark.runner import run_task
from agent.loop import AgentResult
from agent.state import AgentTrajectory, StopReason, TrajectoryEvent


class BenchmarkLoaderTests(unittest.TestCase):
    def test_fixed_split_counts_and_categories(self) -> None:
        train = load_benchmark("train-support")
        heldout = load_benchmark("held-out-eval")
        self.assertEqual((len(train), len(heldout)), (40, 10))
        self.assertEqual(len({task.task_id for task in (*train, *heldout)}), 50)
        for category in {task.category for task in (*train, *heldout)}:
            self.assertEqual(sum(task.category == category for task in train), 4)
            self.assertEqual(sum(task.category == category for task in heldout), 1)
        self.assertEqual(
            {task.category for task in train},
            {
                "repository_navigation",
                "function_discovery",
                "configuration_discovery",
                "environment_variable_discovery",
                "simple_bug_fix",
                "failing_test_fix",
                "validation_change",
                "api_route_fix",
                "configuration_update",
                "explaining_test_failure",
            },
        )

    def test_heldout_ids_are_permanent_and_disjoint_from_training_ids(self) -> None:
        train_ids = set(training_task_ids())
        heldout_ids = {task.task_id for task in load_benchmark("held-out-eval")}
        self.assertEqual(len(heldout_ids), 10)
        self.assertFalse(train_ids & heldout_ids)
        self.assertIn("B005", heldout_ids)
        self.assertIn("B050", heldout_ids)

    def test_unknown_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown benchmark split"):
            load_benchmark("eval")


class BenchmarkRunnerTests(unittest.TestCase):
    def test_task_uses_copy_and_validates_only_that_workspace(self) -> None:
        task = load_benchmark("train-support")[0]
        source_files = {
            path.relative_to(task.snapshot_path): path.read_bytes()
            for path in task.snapshot_path.rglob("*")
            if path.is_file()
        }
        with tempfile.TemporaryDirectory(prefix="benchmark-run-") as parent:
            seen: list[Path] = []

            def fake_agent(instruction: str, workspace: Path) -> AgentResult:
                seen.append(workspace)
                self.assertTrue((workspace / "shop" / "api.py").is_file())
                trajectory = AgentTrajectory(
                    user_request=instruction,
                    started_at="start",
                    events=[TrajectoryEvent(
                        kind="tool", started_at="start", finished_at="finish",
                        tool_name="grep", executed=True,
                    )],
                )
                return AgentResult(
                    final_text="shop",
                    messages=(),
                    tool_calls=(),
                    turns=1,
                    stop_reason=StopReason.COMPLETED,
                    trajectory=trajectory,
                )

            run = run_task(task, fake_agent, workspace_parent=Path(parent))
            self.assertTrue(run.passed)
            self.assertNotEqual(run.workspace, task.snapshot_path)
            self.assertEqual(seen, [run.workspace])
            self.assertEqual(
                source_files,
                {
                    path.relative_to(task.snapshot_path): path.read_bytes()
                    for path in task.snapshot_path.rglob("*")
                    if path.is_file()
                },
            )

    def test_required_tool_is_part_of_success_condition(self) -> None:
        task = load_benchmark("train-support")[0]
        with tempfile.TemporaryDirectory(prefix="benchmark-run-") as parent:
            run = run_task(task, lambda instruction, workspace: "shop api.py", workspace_parent=Path(parent))
            self.assertFalse(run.passed)


if __name__ == "__main__":
    unittest.main()
