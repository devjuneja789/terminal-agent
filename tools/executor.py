"""Validated tool-call dispatch, kept separate from model inference and looping."""

from __future__ import annotations

from typing import Callable

from agent.parser import ToolCall, ToolCallValidationError
from tools.base import ToolResult
from tools.definitions import validate_tool_arguments
from tools.toolbox import Toolbox


class ToolExecutor:
    """Dispatch a validated tool call to the workspace-bound ``Toolbox``."""

    def __init__(self, toolbox: Toolbox) -> None:
        self.toolbox = toolbox
        self._dispatch: dict[str, Callable[..., ToolResult]] = {
            "shell": toolbox.shell,
            "read_file": toolbox.read_file,
            "write_file": toolbox.write_file,
            "grep": toolbox.grep,
            "git": toolbox.git,
        }

    def execute(self, call: ToolCall) -> ToolResult:
        """Revalidate at the execution boundary before invoking any tool."""
        try:
            arguments = validate_tool_arguments(call.name, call.arguments)
        except ValueError as exc:
            raise ToolCallValidationError(str(exc)) from exc
        return self._dispatch[call.name](**arguments)
