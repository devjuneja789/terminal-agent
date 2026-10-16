from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.loop import AgentResult
from agent.state import AgentTrajectory, StopReason, TrajectoryEvent
from evals.manual_tasks import _make_safe_change, _run_test


class ManualSmokeVerifierTests(unittest.TestCase):
    def _result(self, tool_result: dict) -> AgentResult:
        event = TrajectoryEvent(
            kind="tool",
            started_at="start",
            finished_at="finish",
            tool_name=tool_result["tool"],
            executed=True,
            tool_result=tool_result["result"],
        )
        trajectory = AgentTrajectory(user_request="smoke", started_at="start", events=[event])
        return AgentResult(None, (), (), 1, StopReason.COMPLETED, trajectory)

    def test_unittest_summary_on_stderr_counts_as_pass(self) -> None:
        result = self._result({
            "tool": "shell",
            "result": {"success": True, "stdout": "", "stderr": ".\nRan 1 test\n\nOK\n"},
        })
        self.assertTrue(_run_test(result, "")[0])

    def test_exact_requested_marker_without_trailing_newline_counts_as_pass(self) -> None:
        result = self._result({"tool": "write_file", "result": {"success": True}})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "research" / "manual_smoke_marker.txt"
            path.parent.mkdir()
            path.write_text("Terminal-SLM manual smoke task completed.", encoding="utf-8")
            self.assertTrue(_make_safe_change(result, temporary)[0])


if __name__ == "__main__":
    unittest.main()
