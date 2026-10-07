"""Inference-only model client interface.

Tool schemas are provided to inference, but parsing, validation, orchestration,
and tool execution are separate layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, TypedDict


class ChatMessage(TypedDict, total=False):
    role: str
    content: str
    name: str


class ModelClient(ABC):
    """Generate one raw assistant response from the conversation so far."""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]],
        *,
        timeout: float | None = None,
    ) -> str:
        """Run inference and return raw assistant text unchanged.

        Implementations should respect ``timeout`` and raise ``TimeoutError``
        when the remaining wall-clock budget is exhausted.
        """
