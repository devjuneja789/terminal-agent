from __future__ import annotations

import unittest

from evals.benchmark import load_benchmark
from training.agenttune.support_rewards import _contract_evidence_failures, score_support_rollout


class SupportEvidenceTests(unittest.TestCase):
    def test_b016_requires_source_read_from_the_expected_file(self) -> None:
        task = next(task for task in load_benchmark("train-support") if task.task_id == "B016")
        record = {
            "agenttune_trace": {"tool_executions": [
                {"tool": "grep", "schema_valid": True, "arguments": {"pattern": "REQUEST_TIMEOUT_ENV", "path": "shop/config.py"},
                 "result": {"success": True, "stdout": "REQUEST_TIMEOUT_ENV"}},
                {"tool": "read_file", "schema_valid": True, "arguments": {"path": "shop/config.py"},
                 "result": {"success": True, "stdout": "def request_timeout():\\n    return DEFAULT_TIMEOUT_SECONDS\\n"}},
            ]},
        }
        self.assertEqual(_contract_evidence_failures(task, record), [])
        record["agenttune_trace"]["tool_executions"][1]["arguments"]["path"] = "README.md"
        self.assertIn("missing_tool_evidence:read_file", _contract_evidence_failures(task, record))

    def test_expected_nonzero_shell_exit_is_not_failed_execution_penalty(self) -> None:
        task = next(task for task in load_benchmark("train-support") if task.task_id == "B046")
        record = {
            "success": True,
            "support_audit": {
                "initial_validation": {"timed_out": False, "exit_code": 1},
                "changed_files": [],
            },
            "agenttune_trace": {
                "selected_actions": [],
                "parse_errors": [],
                "response_adaptations": [],
                "tool_executions": [
                    {"tool": "shell", "schema_valid": True,
                     "arguments": {"command": "python3 -m unittest test_timeout_from_environment"},
                     "result": {"success": False, "exit_code": 1, "error": None,
                                "stderr": "AssertionError: 10 != 25"}},
                    {"tool": "read_file", "schema_valid": True,
                     "arguments": {"path": "shop/config.py"},
                     "result": {"success": True,
                                "stdout": "def request_timeout():\\n    return DEFAULT_TIMEOUT_SECONDS\\n"}},
                ],
            },
        }
        scored = score_support_rollout(task, record)
        self.assertTrue(scored["benchmark_contract_passed"])
        self.assertEqual(scored["counts"]["failed_tool_executions"], 0)
        self.assertIsNone(scored["reward"])
        record["support_audit"]["changed_files"] = ["shop/config.py"]
        self.assertIn("unexpected_task_file_changes", _contract_evidence_failures(task, record))


if __name__ == "__main__":
    unittest.main()
