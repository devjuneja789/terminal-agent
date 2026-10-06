"""Parse and validate native Qwen3 ``<tool_call>`` assistant output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tools.definitions import validate_tool_arguments

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"


class ToolCallValidationError(ValueError):
    """Raised when native tool-call output is malformed or invalid."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ParsedAssistantResponse:
    """A final assistant response or an ordered batch of validated tool calls."""

    assistant_text: str
    tool_calls: tuple[ToolCall, ...]

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


def _reject_constant(value: str) -> None:
    raise ValueError(f"{value} is not a valid JSON value")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_assistant_response(text: str) -> ParsedAssistantResponse:
    """Parse native Qwen3 tool-call tags and validate every call before return.

    Text with no tool-call markers is a normal final answer. If any marker is
    present, malformed blocks, unknown tool names, and invalid argument objects
    raise ``ToolCallValidationError``. All blocks are validated before the
    agent loop is allowed to execute any of them.
    """
    if not isinstance(text, str):
        raise ToolCallValidationError("assistant response must be text")
    if OPEN_TAG not in text and CLOSE_TAG not in text:
        return ParsedAssistantResponse(assistant_text=text, tool_calls=())

    calls: list[ToolCall] = []
    outside_text: list[str] = []
    cursor = 0
    while cursor < len(text):
        open_at = text.find(OPEN_TAG, cursor)
        close_before = text.find(CLOSE_TAG, cursor)
        if close_before != -1 and (open_at == -1 or close_before < open_at):
            raise ToolCallValidationError("unexpected </tool_call> without an opening <tool_call>")
        if open_at == -1:
            outside_text.append(text[cursor:])
            break

        outside_text.append(text[cursor:open_at])
        payload_start = open_at + len(OPEN_TAG)
        close_at = text.find(CLOSE_TAG, payload_start)
        next_open = text.find(OPEN_TAG, payload_start)
        if close_at == -1:
            raise ToolCallValidationError("unterminated <tool_call> block")
        if next_open != -1 and next_open < close_at:
            raise ToolCallValidationError("nested <tool_call> blocks are not valid")

        payload = text[payload_start:close_at].strip()
        try:
            value = json.loads(
                payload,
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ToolCallValidationError(f"malformed tool-call JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ToolCallValidationError("tool-call JSON must be an object")
        if set(value) != {"name", "arguments"}:
            raise ToolCallValidationError("tool-call JSON must contain exactly 'name' and 'arguments'")
        name = value["name"]
        if not isinstance(name, str):
            raise ToolCallValidationError("tool name must be a string")
        try:
            arguments = validate_tool_arguments(name, value["arguments"])
        except ValueError as exc:
            raise ToolCallValidationError(str(exc)) from exc
        calls.append(ToolCall(name=name, arguments=arguments))
        cursor = close_at + len(CLOSE_TAG)

    return ParsedAssistantResponse(
        assistant_text="".join(outside_text).strip(),
        tool_calls=tuple(calls),
    )
