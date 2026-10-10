from __future__ import annotations

import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from urllib.error import URLError

from cli import EXIT_OK, EXIT_STEP_LIMIT, EXIT_USAGE_ERROR, main
from model.client import ChatMessage, ModelClient
from model.local import LocalOpenAICompatibleClient


def native_call(name: str, arguments: str) -> str:
    return f'<tool_call>{{"name":"{name}","arguments":{arguments}}}</tool_call>'


class ScriptedClient(ModelClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def generate(self, messages, tools, *, timeout=None) -> str:
        return next(self.responses)


class CliTests(unittest.TestCase):
    def test_task_argument_runs_tool_and_prints_actions_and_final_response(self) -> None:
        with tempfile.TemporaryDirectory(prefix="terminal-slm-cli-") as directory:
            workspace = Path(directory) / "task"
            output, errors = io.StringIO(), io.StringIO()
            code = main(
                ["--workspace", str(workspace), "--max-steps", "2", "Write hello", "to a file"],
                model_client=ScriptedClient(
                    [native_call("write_file", '{"path":"hello.txt","content":"hello\\n"}'), "Done."]
                ),
                stdin=io.StringIO(),
                stdout=output,
                stderr=errors,
            )
            self.assertEqual(code, EXIT_OK, errors.getvalue())
            self.assertEqual((workspace / "hello.txt").read_text(), "hello\n")
            self.assertIn("[tool] write_file(", output.getvalue())
            self.assertIn("Final response:\nDone.", output.getvalue())
            self.assertEqual(errors.getvalue(), "")

    def test_default_workspace_is_temporary_and_removed(self) -> None:
        output = io.StringIO()
        code = main(
            ["Say hello"],
            model_client=ScriptedClient(["Hello."]),
            stdin=io.StringIO(),
            stdout=output,
            stderr=io.StringIO(),
        )
        workspace_line = next(line for line in output.getvalue().splitlines() if line.startswith("Workspace: "))
        path = Path(workspace_line.removeprefix("Workspace: "))
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(path.is_relative_to(Path("/tmp")))
        self.assertFalse(path.exists())
        self.assertFalse(path == Path.home())

    def test_task_can_be_read_from_stdin_for_scripts(self) -> None:
        output = io.StringIO()
        code = main(
            [],
            model_client=ScriptedClient(["Completed from stdin."]),
            stdin=io.StringIO("Do the task from stdin\n"),
            stdout=output,
            stderr=io.StringIO(),
        )
        self.assertEqual(code, EXIT_OK)
        self.assertIn("Completed from stdin.", output.getvalue())

    def test_max_steps_has_distinct_nonzero_exit_code(self) -> None:
        output, errors = io.StringIO(), io.StringIO()
        code = main(
            ["--max-steps", "0", "Use a tool"],
            model_client=ScriptedClient([native_call("shell", '{"command":"pwd"}')]),
            stdin=io.StringIO(),
            stdout=output,
            stderr=errors,
        )
        self.assertEqual(code, EXIT_STEP_LIMIT)
        self.assertIn("skipped", output.getvalue())
        self.assertIn("maximum_tool_calls", errors.getvalue())

    def test_missing_task_returns_usage_error(self) -> None:
        errors = io.StringIO()
        code = main([], stdin=io.StringIO("  \n"), stdout=io.StringIO(), stderr=errors)
        self.assertEqual(code, EXIT_USAGE_ERROR)
        self.assertIn("provide a task", errors.getvalue())

    def test_loopback_endpoint_validation_rejects_remote_hosts(self) -> None:
        for endpoint in ("https://example.com/v1/chat/completions", "file:///etc/passwd"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                LocalOpenAICompatibleClient(endpoint)

    def test_local_endpoint_timeout_preserves_model_client_timeout_contract(self) -> None:
        client = LocalOpenAICompatibleClient("http://127.0.0.1:8080/v1/chat/completions")
        with patch("model.local.urlopen", side_effect=URLError(TimeoutError("deadline"))):
            with self.assertRaises(TimeoutError):
                client.generate([], [], timeout=0.1)

    def test_local_endpoint_runs_through_native_tool_call_parser(self) -> None:
        requests: list[dict] = []
        responses = [
            {"choices": [{"message": {"tool_calls": [{"function": {
                "name": "write_file",
                "arguments": '{"path":"endpoint.txt","content":"from local server\\n"}',
            }}]}}]},
            {"choices": [{"message": {"content": "Endpoint task completed."}}]},
        ]
        from io import BytesIO

        def fake_urlopen(request, timeout=None):
            requests.append(json.loads(request.data.decode("utf-8")))
            return BytesIO(json.dumps(responses.pop(0)).encode("utf-8"))

        with tempfile.TemporaryDirectory(prefix="terminal-slm-endpoint-") as directory:
            workspace = Path(directory) / "workspace"
            output = io.StringIO()
            endpoint = "http://127.0.0.1:8080/v1/chat/completions"
            with patch("model.local.urlopen", side_effect=fake_urlopen):
                code = main(
                    ["--model-endpoint", endpoint, "--workspace", str(workspace), "Write a file"],
                    stdin=io.StringIO(),
                    stdout=output,
                    stderr=io.StringIO(),
                )
            self.assertEqual(code, EXIT_OK, output.getvalue())
            self.assertEqual((workspace / "endpoint.txt").read_text(), "from local server\n")
            self.assertIn("Endpoint task completed.", output.getvalue())
            self.assertEqual(requests[0]["model"], "Qwen3-1.7B")
            self.assertEqual(len(requests[0]["tools"]), 5)


if __name__ == "__main__":
    unittest.main()
