"""Minimal model/tool orchestration for Phase A2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from agent.parser import ToolCall, parse_assistant_response
from model.client import ChatMessage, ModelClient
from tools.definitions import get_tool_schemas
from tools.executor import ToolExecutor

DEFAULT_SYSTEM_PROMPT = (
    "You are a terminal coding assistant. Use the provided tools when needed. "
    "For a tool call, emit Qwen3's native format exactly as "
    '<tool_call>{"name":"tool_name","arguments":{}}</tool_call>. '
    "When the task is complete, answer normally without a tool-call block."
)


@dataclass(frozen=True)
class AgentResult:
    final_text: str
    messages: tuple[ChatMessage, ...]
    tool_calls: tuple[ToolCall, ...]
    turns: int


class AgentTurnLimitError(RuntimeError):
    """Raised when the model keeps requesting tools beyond the turn limit."""


class AgentLoop:
    """Connect raw model inference, response validation, and tool execution."""

    def __init__(
        self,
        model_client: ModelClient,
        tool_executor: ToolExecutor,
        *,
        max_turns: int = 12,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.model_client = model_client
        self.tool_executor = tool_executor
        self.max_turns = max_turns
        self.tool_schemas = get_tool_schemas()

    def run(
        self,
        user_message: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> AgentResult:
        """Run turns until the model returns normal assistant text.

        Every response is fully parsed and validated before any call in that
        response is executed. Tool results are added as native ``tool`` role
        messages, and the next inference receives the full conversation.
        """
        messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        executed_calls: list[ToolCall] = []

        for turn in range(1, self.max_turns + 1):
            raw_response = self.model_client.generate(messages, self.tool_schemas)
            parsed = parse_assistant_response(raw_response)
            messages.append({"role": "assistant", "content": raw_response})
            if parsed.is_final:
                return AgentResult(
                    final_text=parsed.assistant_text,
                    messages=tuple(messages),
                    tool_calls=tuple(executed_calls),
                    turns=turn,
                )

            # Parser validation covers every call before this loop can execute
            # the first one, so an invalid later block cannot partially run.
            for call in parsed.tool_calls:
                result = self.tool_executor.execute(call)
                executed_calls.append(call)
                messages.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "content": json.dumps(result.as_dict(), ensure_ascii=False),
                    }
                )

        raise AgentTurnLimitError(f"agent exceeded the maximum of {self.max_turns} model turns")
