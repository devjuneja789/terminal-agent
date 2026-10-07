"""Conversation state and compact, timestamped agent trajectory records."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from model.client import ChatMessage


class StopReason(str, Enum):
    COMPLETED = "completed"
    MAX_TOOL_CALLS = "maximum_tool_calls"
    EXECUTION_TIMEOUT = "execution_timeout"
    CONVERSATION_LIMIT = "conversation_limit"
    INVALID_TOOL_CALL = "invalid_tool_call"
    MODEL_ERROR = "model_error"


@dataclass(frozen=True)
class AgentConfig:
    """Resource limits for one task run."""

    max_tool_calls: int = 16
    max_execution_time_seconds: float = 120.0
    max_conversation_chars: int = 100_000
    max_tool_output_chars: int = 16_000

    def __post_init__(self) -> None:
        if not isinstance(self.max_tool_calls, int) or isinstance(self.max_tool_calls, bool):
            raise ValueError("max_tool_calls must be an integer")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must not be negative")
        if not math.isfinite(self.max_execution_time_seconds) or self.max_execution_time_seconds <= 0:
            raise ValueError("max_execution_time_seconds must be finite and positive")
        if not isinstance(self.max_conversation_chars, int) or isinstance(self.max_conversation_chars, bool):
            raise ValueError("max_conversation_chars must be an integer")
        if self.max_conversation_chars < 256:
            raise ValueError("max_conversation_chars must be at least 256")
        if not isinstance(self.max_tool_output_chars, int) or isinstance(self.max_tool_output_chars, bool):
            raise ValueError("max_tool_output_chars must be an integer")
        if self.max_tool_output_chars < 256:
            raise ValueError("max_tool_output_chars must be at least 256")


@dataclass
class TrajectoryEvent:
    kind: str
    started_at: str
    finished_at: str | None = None
    model_output: str | None = None
    model_output_truncated: bool = False
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    executed: bool | None = None
    tool_result: dict[str, Any] | None = None
    tool_result_truncated: bool = False
    error: str | None = None


@dataclass
class AgentTrajectory:
    user_request: str
    started_at: str
    events: list[TrajectoryEvent] = field(default_factory=list)
    final_response: str | None = None
    finished_at: str | None = None
    stop_reason: str | None = None

    @staticmethod
    def timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def start_event(
        self,
        kind: str,
        *,
        tool_name: str | None = None,
        tool_arguments: dict[str, Any] | None = None,
    ) -> TrajectoryEvent:
        event = TrajectoryEvent(
            kind=kind,
            started_at=self.timestamp(),
            tool_name=tool_name,
            tool_arguments=tool_arguments,
        )
        self.events.append(event)
        return event

    def finish_event(self, event: TrajectoryEvent) -> None:
        event.finished_at = self.timestamp()

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_request": self.user_request,
            "started_at": self.started_at,
            "events": [asdict(event) for event in self.events],
            "final_response": self.final_response,
            "finished_at": self.finished_at,
            "stop_reason": self.stop_reason,
        }


@dataclass
class AgentState:
    messages: list[ChatMessage]
    trajectory: AgentTrajectory

    @classmethod
    def create(cls, *, user_message: str, system_prompt: str) -> AgentState:
        now = AgentTrajectory.timestamp()
        trajectory = AgentTrajectory(user_request=user_message, started_at=now)
        messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return cls(messages=messages, trajectory=trajectory)

    def conversation_chars(self, additional: ChatMessage | None = None) -> int:
        messages = self.messages if additional is None else [*self.messages, additional]
        return sum(
            len(message.get("role", ""))
            + len(message.get("content", ""))
            + len(message.get("name", ""))
            for message in messages
        )
