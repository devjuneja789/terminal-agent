"""Bounded, deterministic model/tool orchestration for Phase A3."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from agent.parser import ToolCall, ToolCallValidationError, parse_assistant_response
from agent.prompts import DEFAULT_SYSTEM_PROMPT
from agent.state import AgentConfig, AgentState, AgentTrajectory, StopReason
from model.client import ChatMessage, ModelClient
from tools.base import ToolResult
from tools.definitions import get_tool_schemas
from tools.executor import ToolExecutor


@dataclass(frozen=True)
class AgentResult:
    """Outcome plus the full bounded conversation and execution trajectory."""

    final_text: str | None
    messages: tuple[ChatMessage, ...]
    tool_calls: tuple[ToolCall, ...]
    turns: int
    stop_reason: StopReason
    trajectory: AgentTrajectory

    @property
    def completed(self) -> bool:
        return self.stop_reason == StopReason.COMPLETED


class AgentLoop:
    """Run model turns and workspace tools until completion or a configured limit."""

    def __init__(
        self,
        model_client: ModelClient,
        tool_executor: ToolExecutor,
        *,
        config: AgentConfig | None = None,
    ) -> None:
        self.model_client = model_client
        self.tool_executor = tool_executor
        self.config = config or AgentConfig()
        self.tool_schemas = get_tool_schemas()

    def run(
        self,
        user_message: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> AgentResult:
        """Run a task, preserving model and tool events in a timestamped trace."""
        state = AgentState.create(user_message=user_message, system_prompt=system_prompt)
        deadline = time.monotonic() + self.config.max_execution_time_seconds
        executed_calls: list[ToolCall] = []
        turns = 0

        if state.conversation_chars() > self.config.max_conversation_chars:
            return self._finish(state, executed_calls, turns, StopReason.CONVERSATION_LIMIT)

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._finish(state, executed_calls, turns, StopReason.EXECUTION_TIMEOUT)

            model_event = state.trajectory.start_event("model")
            turns += 1
            try:
                raw_response = self.model_client.generate(
                    state.messages,
                    self.tool_schemas,
                    timeout=remaining,
                )
            except TimeoutError as exc:
                model_event.error = str(exc) or "model inference timed out"
                state.trajectory.finish_event(model_event)
                return self._finish(state, executed_calls, turns, StopReason.EXECUTION_TIMEOUT)
            except Exception as exc:
                model_event.error = f"model inference failed: {exc}"
                state.trajectory.finish_event(model_event)
                return self._finish(state, executed_calls, turns, StopReason.MODEL_ERROR)

            if not isinstance(raw_response, str):
                model_event.error = "model client returned a non-text response"
                state.trajectory.finish_event(model_event)
                return self._finish(state, executed_calls, turns, StopReason.MODEL_ERROR)

            if len(raw_response) > self.config.max_conversation_chars:
                model_event.model_output = raw_response[: self.config.max_conversation_chars]
                model_event.model_output_truncated = True
                state.trajectory.finish_event(model_event)
                return self._finish(state, executed_calls, turns, StopReason.CONVERSATION_LIMIT)

            model_event.model_output = raw_response
            state.trajectory.finish_event(model_event)
            if time.monotonic() >= deadline:
                return self._finish(state, executed_calls, turns, StopReason.EXECUTION_TIMEOUT)

            assistant_message: ChatMessage = {"role": "assistant", "content": raw_response}
            if state.conversation_chars(assistant_message) > self.config.max_conversation_chars:
                return self._finish(state, executed_calls, turns, StopReason.CONVERSATION_LIMIT)
            state.messages.append(assistant_message)

            try:
                parsed = parse_assistant_response(raw_response)
            except ToolCallValidationError as exc:
                model_event.error = str(exc)
                return self._finish(state, executed_calls, turns, StopReason.INVALID_TOOL_CALL)

            if parsed.is_final:
                state.trajectory.final_response = parsed.assistant_text
                return self._finish(state, executed_calls, turns, StopReason.COMPLETED)

            for call in parsed.tool_calls:
                tool_event = state.trajectory.start_event(
                    "tool",
                    tool_name=call.name,
                    tool_arguments=dict(call.arguments),
                )
                if len(executed_calls) >= self.config.max_tool_calls:
                    tool_event.executed = False
                    tool_event.error = "maximum tool-call limit reached; tool was not executed"
                    state.trajectory.finish_event(tool_event)
                    return self._finish(state, executed_calls, turns, StopReason.MAX_TOOL_CALLS)

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    tool_event.executed = False
                    tool_event.error = "execution time limit reached; tool was not executed"
                    state.trajectory.finish_event(tool_event)
                    return self._finish(state, executed_calls, turns, StopReason.EXECUTION_TIMEOUT)

                available_chars = (
                    self.config.max_conversation_chars
                    - state.conversation_chars()
                    - len("tool")
                    - len(call.name)
                )
                if available_chars < 256:
                    tool_event.executed = False
                    tool_event.error = "conversation size limit reached; tool was not executed"
                    state.trajectory.finish_event(tool_event)
                    return self._finish(state, executed_calls, turns, StopReason.CONVERSATION_LIMIT)

                observation_limit = min(self.config.max_tool_output_chars, available_chars)
                try:
                    result = self.tool_executor.execute(call, timeout_budget=remaining)
                except Exception as exc:
                    result = ToolResult(success=False, error=f"tool execution failed: {exc}")

                observation, encoded, was_truncated = _bounded_tool_result(result, observation_limit)
                tool_event.executed = True
                tool_event.tool_result = observation
                tool_event.tool_result_truncated = was_truncated
                state.trajectory.finish_event(tool_event)
                executed_calls.append(call)

                tool_message: ChatMessage = {
                    "role": "tool",
                    "name": call.name,
                    "content": encoded,
                }
                state.messages.append(tool_message)

                if time.monotonic() >= deadline:
                    return self._finish(state, executed_calls, turns, StopReason.EXECUTION_TIMEOUT)

    @staticmethod
    def _finish(
        state: AgentState,
        executed_calls: list[ToolCall],
        turns: int,
        reason: StopReason,
    ) -> AgentResult:
        state.trajectory.stop_reason = reason.value
        state.trajectory.finished_at = AgentTrajectory.timestamp()
        return AgentResult(
            final_text=state.trajectory.final_response,
            messages=tuple(state.messages),
            tool_calls=tuple(executed_calls),
            turns=turns,
            stop_reason=reason,
            trajectory=state.trajectory,
        )


def _bounded_tool_result(result: ToolResult, max_chars: int) -> tuple[dict[str, Any], str, bool]:
    """Serialize a tool result within a character budget, marking clipped fields."""
    payload = result.as_dict()
    truncated = False
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= max_chars:
        return payload, encoded, truncated

    payload["metadata"] = {}
    truncated = True
    text_fields = ["stdout", "stderr", "error"]
    for field in text_fields:
        if not isinstance(payload.get(field), str):
            payload[field] = ""
    payload["stdout_truncated"] = bool(payload.get("stdout_truncated"))
    payload["stderr_truncated"] = bool(payload.get("stderr_truncated"))

    while True:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= max_chars:
            return payload, encoded, truncated
        excess = len(encoded) - max_chars
        candidate = max(text_fields, key=lambda field: len(payload[field]))
        current = payload[candidate]
        if not current:
            minimal = {
                "success": bool(result.success),
                "stdout": "",
                "stderr": "",
                "exit_code": result.exit_code,
                "error": None,
                "stdout_truncated": True,
                "stderr_truncated": True,
                "metadata": {},
            }
            minimal_encoded = json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))
            return minimal, minimal_encoded, True
        remove_count = min(len(current), max(1, (excess + 1) // 2))
        payload[candidate] = current[:-remove_count]
        if candidate == "stdout":
            payload["stdout_truncated"] = True
        elif candidate == "stderr":
            payload["stderr_truncated"] = True
