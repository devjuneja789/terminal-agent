from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from agent.loop import AgentLoop
from agent.parser import ToolCallValidationError, parse_assistant_response
from model.client import ChatMessage, ModelClient
from model.local import LocalTransformersClient
from tools import Toolbox
from tools.definitions import get_tool_schemas
from tools.executor import ToolExecutor


def native_call(name: str, arguments: str) -> str:
    return f'<tool_call>{{"name":"{name}","arguments":{arguments}}}</tool_call>'


class FakeModelClient(ModelClient):
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[ChatMessage], list[dict[str, Any]]]] = []

    def generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]],
    ) -> str:
        self.calls.append((list(messages), list(tools)))
        if not self.responses:
            raise AssertionError("fake model received more generation calls than expected")
        return self.responses.pop(0)


class NativeToolCallParserTests(unittest.TestCase):
    def test_local_client_is_lazy_and_does_not_request_downloads_by_default(self) -> None:
        client = LocalTransformersClient()
        self.assertTrue(client.local_files_only)
        self.assertIsNone(client._model)
        self.assertIsNone(client._tokenizer)
        with self.assertRaises(ValueError):
            LocalTransformersClient(max_new_tokens=0)

    def test_tool_definitions_use_standard_openai_function_schemas(self) -> None:
        schemas = get_tool_schemas()
        self.assertEqual(
            [tool["function"]["name"] for tool in schemas],
            ["shell", "read_file", "write_file", "grep", "git"],
        )
        self.assertTrue(all(tool["type"] == "function" for tool in schemas))
        self.assertTrue(all("parameters" in tool["function"] for tool in schemas))

    def test_valid_native_tool_call(self) -> None:
        parsed = parse_assistant_response(native_call("write_file", '{"path":"note.txt","content":"hello"}'))
        self.assertFalse(parsed.is_final)
        self.assertEqual(parsed.tool_calls[0].name, "write_file")
        self.assertEqual(parsed.tool_calls[0].arguments, {"path": "note.txt", "content": "hello"})

    def test_malformed_tool_call_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolCallValidationError, "malformed tool-call JSON"):
            parse_assistant_response('<tool_call>{"name":"shell","arguments":}</tool_call>')
        with self.assertRaisesRegex(ToolCallValidationError, "unterminated"):
            parse_assistant_response('<tool_call>{"name":"shell","arguments":{"command":"pwd"}}')

    def test_unknown_tool_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolCallValidationError, "unknown tool"):
            parse_assistant_response(native_call("open_browser", "{}"))

    def test_invalid_and_missing_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ToolCallValidationError, "must be string"):
            parse_assistant_response(native_call("shell", '{"command":123}'))
        with self.assertRaisesRegex(ToolCallValidationError, "missing required"):
            parse_assistant_response(native_call("write_file", '{"path":"note.txt"}'))
        with self.assertRaisesRegex(ToolCallValidationError, "end_line"):
            parse_assistant_response(native_call("read_file", '{"path":"note.txt","start_line":5,"end_line":2}'))
        with self.assertRaisesRegex(ToolCallValidationError, "workspace-relative"):
            parse_assistant_response(native_call("read_file", '{"path":"/etc/passwd"}'))
        with self.assertRaisesRegex(ToolCallValidationError, "parent directory"):
            parse_assistant_response(native_call("write_file", '{"path":"../outside","content":"no"}'))
        with self.assertRaisesRegex(ToolCallValidationError, "must not be blank"):
            parse_assistant_response(native_call("shell", '{"command":"   "}'))

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ToolCallValidationError, "duplicate JSON key"):
            parse_assistant_response(
                '<tool_call>{"name":"shell","arguments":{"command":"pwd","command":"id"}}</tool_call>'
            )

    def test_normal_final_response_is_not_a_tool_call(self) -> None:
        text = "The requested file is already correct."
        parsed = parse_assistant_response(text)
        self.assertTrue(parsed.is_final)
        self.assertEqual(parsed.assistant_text, text)


class AgentLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="terminal-slm-agent-test-")
        self.workspace = Path(self.temporary.name) / "workspace"
        self.workspace.mkdir()
        self.toolbox = Toolbox(self.workspace)

    def tearDown(self) -> None:
        self.toolbox.close()
        self.temporary.cleanup()

    def test_multiple_sequential_tool_calls_then_final_response(self) -> None:
        model = FakeModelClient(
            [
                native_call("write_file", '{"path":"src/message.txt","content":"ready\\n"}'),
                native_call("read_file", '{"path":"src/message.txt"}'),
                "The file contains: ready",
            ]
        )
        loop = AgentLoop(model, ToolExecutor(self.toolbox))

        result = loop.run("Create and check a message file.")

        self.assertEqual(result.final_text, "The file contains: ready")
        self.assertEqual([call.name for call in result.tool_calls], ["write_file", "read_file"])
        self.assertEqual(result.turns, 3)
        self.assertEqual((self.workspace / "src/message.txt").read_text(), "ready\n")
        self.assertEqual([message["role"] for message in result.messages[-5:]], [
            "assistant", "tool", "assistant", "tool", "assistant"
        ])
        self.assertEqual(model.calls[0][1], get_tool_schemas())
        self.assertIn("ready", model.calls[2][0][-1]["content"])

    def test_ordered_multiple_calls_in_one_response_execute_sequentially(self) -> None:
        combined = (
            native_call("write_file", '{"path":"step.txt","content":"one"}')
            + native_call("read_file", '{"path":"step.txt"}')
        )
        model = FakeModelClient([combined, "Done."])
        result = AgentLoop(model, ToolExecutor(self.toolbox)).run("Write then read.")
        self.assertEqual([call.name for call in result.tool_calls], ["write_file", "read_file"])
        self.assertEqual(result.turns, 2)

    def test_invalid_later_call_prevents_all_execution_for_that_response(self) -> None:
        response = (
            native_call("write_file", '{"path":"must-not-exist.txt","content":"no"}')
            + native_call("shell", '{"command":123}')
        )
        model = FakeModelClient([response])
        loop = AgentLoop(model, ToolExecutor(self.toolbox))
        with self.assertRaises(ToolCallValidationError):
            loop.run("Try the calls.")
        self.assertFalse((self.workspace / "must-not-exist.txt").exists())

    def test_normal_final_response_ends_without_tool_execution(self) -> None:
        model = FakeModelClient(["No tool is needed."])
        result = AgentLoop(model, ToolExecutor(self.toolbox)).run("Say whether a tool is needed.")
        self.assertEqual(result.final_text, "No tool is needed.")
        self.assertEqual(result.tool_calls, ())
        self.assertEqual(result.turns, 1)


if __name__ == "__main__":
    unittest.main()
