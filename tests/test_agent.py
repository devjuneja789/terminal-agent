from __future__ import annotations

import json
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from agent.loop import AgentLoop
from agent.parser import ToolCallValidationError, parse_assistant_response
from agent.state import AgentConfig, StopReason
from model.client import ChatMessage, ModelClient
from model.local import LocalTransformersClient
from tools import Toolbox
from tools.definitions import get_tool_schemas
from tools.executor import ToolExecutor


def native_call(name: str, arguments: str) -> str:
    return f'<tool_call>{{"name":"{name}","arguments":{arguments}}}</tool_call>'


class MockModelClient(ModelClient):
    """Deterministic scripted client; no model packages or weights are needed."""

    def __init__(self, responses: Sequence[str], *, delay_seconds: float = 0.0) -> None:
        self.responses = list(responses)
        self.delay_seconds = delay_seconds
        self.calls: list[tuple[list[ChatMessage], list[dict[str, Any]], float | None]] = []

    def generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]],
        *,
        timeout: float | None = None,
    ) -> str:
        self.calls.append((list(messages), list(tools), timeout))
        if self.delay_seconds:
            if timeout is not None and self.delay_seconds > timeout:
                time.sleep(timeout)
                raise TimeoutError("mock inference timed out")
            time.sleep(self.delay_seconds)
        if not self.responses:
            raise AssertionError("mock model received more generation calls than expected")
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
        self.executor = ToolExecutor(self.toolbox)

    def tearDown(self) -> None:
        self.toolbox.close()
        self.temporary.cleanup()

    def test_simple_final_answer(self) -> None:
        model = MockModelClient(["No tool is needed."])
        result = AgentLoop(model, self.executor).run("Say whether a tool is needed.")
        self.assertTrue(result.completed)
        self.assertEqual(result.final_text, "No tool is needed.")
        self.assertEqual(result.tool_calls, ())
        self.assertEqual(result.turns, 1)

    def test_one_tool_call_then_final_answer_records_complete_trajectory(self) -> None:
        model = MockModelClient(
            [
                native_call("write_file", '{"path":"answer.txt","content":"42\\n"}'),
                "I wrote the answer file.",
            ]
        )
        result = AgentLoop(model, self.executor).run("Write 42 to answer.txt.")
        self.assertTrue(result.completed)
        self.assertEqual((self.workspace / "answer.txt").read_text(), "42\n")
        trajectory = result.trajectory
        self.assertEqual(trajectory.user_request, "Write 42 to answer.txt.")
        self.assertEqual(trajectory.final_response, "I wrote the answer file.")
        self.assertIsNotNone(trajectory.finished_at)
        datetime.fromisoformat(trajectory.started_at)
        datetime.fromisoformat(trajectory.finished_at)
        self.assertEqual([event.kind for event in trajectory.events], ["model", "tool", "model"])
        model_event, tool_event, final_event = trajectory.events
        self.assertIn("<tool_call>", model_event.model_output or "")
        self.assertIsNotNone(model_event.started_at)
        self.assertIsNotNone(model_event.finished_at)
        datetime.fromisoformat(model_event.started_at)
        datetime.fromisoformat(model_event.finished_at)
        self.assertEqual(tool_event.tool_name, "write_file")
        self.assertEqual(tool_event.tool_arguments, {"path": "answer.txt", "content": "42\n"})
        self.assertTrue(tool_event.executed)
        self.assertTrue(tool_event.tool_result["success"])
        self.assertEqual(final_event.model_output, "I wrote the answer file.")
        self.assertEqual(trajectory.stop_reason, StopReason.COMPLETED)
        json.dumps(trajectory.to_dict())

    def test_multiple_tool_calls_are_sequential_and_observed(self) -> None:
        model = MockModelClient(
            [
                native_call("write_file", '{"path":"src/message.txt","content":"ready\\n"}'),
                native_call("read_file", '{"path":"src/message.txt"}'),
                "The file contains: ready",
            ]
        )
        result = AgentLoop(model, self.executor).run("Create and check a message file.")
        self.assertEqual(result.final_text, "The file contains: ready")
        self.assertEqual([call.name for call in result.tool_calls], ["write_file", "read_file"])
        self.assertEqual(result.turns, 3)
        self.assertEqual((self.workspace / "src/message.txt").read_text(), "ready\n")
        self.assertEqual([message["role"] for message in result.messages[-5:]], [
            "assistant", "tool", "assistant", "tool", "assistant"
        ])
        self.assertIn("ready", model.calls[2][0][-1]["content"])
        self.assertTrue(all(model_call[1] == get_tool_schemas() for model_call in model.calls))

    def test_ordered_multiple_calls_in_one_response_execute_sequentially(self) -> None:
        combined = (
            native_call("write_file", '{"path":"step.txt","content":"one"}')
            + native_call("read_file", '{"path":"step.txt"}')
        )
        model = MockModelClient([combined, "Done."])
        result = AgentLoop(model, self.executor).run("Write then read.")
        self.assertEqual([call.name for call in result.tool_calls], ["write_file", "read_file"])
        self.assertEqual(result.turns, 2)

    def test_malformed_tool_call_stops_without_execution(self) -> None:
        model = MockModelClient([native_call("write_file", '{"path":"bad.txt","content":}')])
        result = AgentLoop(model, self.executor).run("Try a malformed call.")
        self.assertEqual(result.stop_reason, StopReason.INVALID_TOOL_CALL)
        self.assertFalse((self.workspace / "bad.txt").exists())
        self.assertIn("malformed tool-call JSON", result.trajectory.events[0].error or "")

    def test_invalid_later_call_prevents_prior_call_execution(self) -> None:
        response = (
            native_call("write_file", '{"path":"must-not-exist.txt","content":"no"}')
            + native_call("shell", '{"command":123}')
        )
        model = MockModelClient([response])
        result = AgentLoop(model, self.executor).run("Try the calls.")
        self.assertEqual(result.stop_reason, StopReason.INVALID_TOOL_CALL)
        self.assertFalse((self.workspace / "must-not-exist.txt").exists())

    def test_tool_failure_is_returned_as_observation_and_loop_continues(self) -> None:
        model = MockModelClient([native_call("read_file", '{"path":"missing.txt"}'), "I could not read it."])
        result = AgentLoop(model, self.executor).run("Read missing.txt.")
        self.assertTrue(result.completed)
        tool_event = next(event for event in result.trajectory.events if event.kind == "tool")
        self.assertFalse(tool_event.tool_result["success"])
        self.assertIn("not a file", tool_event.tool_result["error"])
        self.assertEqual(result.final_text, "I could not read it.")

    def test_maximum_tool_calls_terminates_before_extra_execution(self) -> None:
        model = MockModelClient(
            [
                native_call("write_file", '{"path":"first.txt","content":"1"}'),
                native_call("write_file", '{"path":"second.txt","content":"2"}'),
            ]
        )
        config = AgentConfig(max_tool_calls=1)
        result = AgentLoop(model, self.executor, config=config).run("Write two files.")
        self.assertEqual(result.stop_reason, StopReason.MAX_TOOL_CALLS)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertTrue((self.workspace / "first.txt").exists())
        self.assertFalse((self.workspace / "second.txt").exists())
        skipped = result.trajectory.events[-1]
        self.assertFalse(skipped.executed)
        self.assertIn("limit reached", skipped.error or "")

    def test_execution_timeout_passes_remaining_budget_and_stops(self) -> None:
        model = MockModelClient(["This should not be accepted as a late final."], delay_seconds=0.05)
        config = AgentConfig(max_execution_time_seconds=0.005)
        result = AgentLoop(model, self.executor, config=config).run("Wait for a slow model.")
        self.assertEqual(result.stop_reason, StopReason.EXECUTION_TIMEOUT)
        self.assertIsNone(result.final_text)
        self.assertEqual(len(model.calls), 1)
        self.assertLessEqual(model.calls[0][2], config.max_execution_time_seconds)

    def test_tool_output_and_conversation_sizes_are_bounded(self) -> None:
        model = MockModelClient(
            [native_call("shell", '{"command":"printf \'%01000d\' 1"}'), "Output was capped."]
        )
        config = AgentConfig(max_tool_output_chars=256, max_conversation_chars=4096)
        result = AgentLoop(model, self.executor, config=config).run("Print a lot of output.")
        tool_message = next(message for message in result.messages if message["role"] == "tool")
        self.assertLessEqual(len(tool_message["content"]), config.max_tool_output_chars)
        tool_event = next(event for event in result.trajectory.events if event.kind == "tool")
        self.assertTrue(tool_event.tool_result_truncated)
        self.assertTrue(result.completed)

    def test_oversized_initial_conversation_stops_before_inference(self) -> None:
        model = MockModelClient(["unused"])
        config = AgentConfig(max_conversation_chars=256)
        result = AgentLoop(model, self.executor, config=config).run("task", system_prompt="s" * 300)
        self.assertEqual(result.stop_reason, StopReason.CONVERSATION_LIMIT)
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
