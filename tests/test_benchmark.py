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
                    final_text="shop/api.py",
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

    def test_read_file_evidence_checks_arguments_and_observed_content(self) -> None:
        task = next(task for task in load_benchmark("train-support") if task.task_id == "B016")
        with tempfile.TemporaryDirectory(prefix="benchmark-run-") as parent:
            def fake_agent(instruction: str, workspace: Path) -> AgentResult:
                trajectory = AgentTrajectory(
                    user_request=instruction,
                    started_at="start",
                    events=[
                        TrajectoryEvent(
                            kind="tool", started_at="start", finished_at="finish",
                            tool_name="grep", tool_arguments={"pattern": "REQUEST_TIMEOUT_ENV", "path": "shop/config.py"},
                            executed=True, tool_result={"success": True, "stdout": "shop/config.py:8:REQUEST_TIMEOUT_ENV = \\\"REQUEST_TIMEOUT_SECONDS\\\""},
                        ),
                        TrajectoryEvent(
                            kind="tool", started_at="start", finished_at="finish",
                            tool_name="read_file", tool_arguments={"path": "shop/config.py", "start_line": 8, "end_line": 13},
                            executed=True, tool_result={"success": True, "stdout": "def request_timeout():\\n    return DEFAULT_TIMEOUT_SECONDS\\n"},
                        ),
                    ],
                )
                return AgentResult(
                    final_text="REQUEST_TIMEOUT_SECONDS is declared, but request_timeout() does not read it and returns DEFAULT_TIMEOUT_SECONDS.",
                    messages=(), tool_calls=(), turns=3, stop_reason=StopReason.COMPLETED,
                    trajectory=trajectory,
                )

            run = run_task(task, fake_agent, workspace_parent=Path(parent))
            self.assertTrue(run.passed, run.criterion_failures)

    def test_read_file_evidence_rejects_wrong_path(self) -> None:
        task = next(task for task in load_benchmark("train-support") if task.task_id == "B016")
        with tempfile.TemporaryDirectory(prefix="benchmark-run-") as parent:
            def fake_agent(instruction: str, workspace: Path) -> AgentResult:
                trajectory = AgentTrajectory(
                    user_request=instruction, started_at="start", events=[
                        TrajectoryEvent(
                            kind="tool", started_at="start", finished_at="finish",
                            tool_name="grep", tool_arguments={"pattern": "REQUEST_TIMEOUT_ENV", "path": "shop/config.py"},
                            executed=True, tool_result={"success": True, "stdout": "REQUEST_TIMEOUT_ENV"},
                        ),
                        TrajectoryEvent(
                            kind="tool", started_at="start", finished_at="finish",
                            tool_name="read_file", tool_arguments={"path": "README.md", "start_line": 1, "end_line": 10},
                            executed=True, tool_result={"success": True, "stdout": "def request_timeout():\\n    return DEFAULT_TIMEOUT_SECONDS\\n"},
                        ),
                    ],
                )
                return AgentResult(
                    final_text="REQUEST_TIMEOUT_SECONDS is declared, but request_timeout() does not read it and returns DEFAULT_TIMEOUT_SECONDS.",
                    messages=(), tool_calls=(), turns=3, stop_reason=StopReason.COMPLETED,
                    trajectory=trajectory,
                )

            run = run_task(task, fake_agent, workspace_parent=Path(parent))
            self.assertFalse(run.passed)
            self.assertIn("missing_tool_evidence:read_file", run.criterion_failures)

    def test_read_only_explanation_task_rejects_workspace_changes(self) -> None:
        task = next(task for task in load_benchmark("train-support") if task.task_id == "B046")
        with tempfile.TemporaryDirectory(prefix="benchmark-run-") as parent:
            def fake_agent(instruction: str, workspace: Path) -> AgentResult:
                (workspace / "shop/config.py").write_text(
                    (workspace / "shop/config.py").read_text(encoding="utf-8") + "\\n# unrelated edit\\n",
                    encoding="utf-8",
                )
                trajectory = AgentTrajectory(
                    user_request=instruction, started_at="start", events=[
                        TrajectoryEvent(
                            kind="tool", started_at="start", finished_at="finish",
                            tool_name="shell", tool_arguments={"command": "python3 -m unittest test_timeout_from_environment"},
                            executed=True, tool_result={"success": False, "exit_code": 1, "error": None,
                                                        "stderr": "AssertionError: 10 != 25"},
                        ),
                        TrajectoryEvent(
                            kind="tool", started_at="start", finished_at="finish",
                            tool_name="read_file", tool_arguments={"path": "shop/config.py"},
                            executed=True, tool_result={"success": True,
                                                        "stdout": "def request_timeout():\\n    return DEFAULT_TIMEOUT_SECONDS\\n"},
                        ),
                    ],
                )
                return AgentResult(
                    final_text="The test fails with AssertionError: 10 != 25 because request_timeout does not read REQUEST_TIMEOUT_SECONDS and returns DEFAULT_TIMEOUT_SECONDS.",
                    messages=(), tool_calls=(), turns=3, stop_reason=StopReason.COMPLETED,
                    trajectory=trajectory,
                )

            run = run_task(task, fake_agent, workspace_parent=Path(parent))
            self.assertFalse(run.passed)
            self.assertIn("unexpected_task_file_changes", run.criterion_failures)
            self.assertEqual(run.changed_files, ("shop/config.py",))


if __name__ == "__main__":
    unittest.main()
