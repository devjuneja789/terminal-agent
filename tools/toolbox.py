"""Public interface binding the five agent tools to one workspace."""

from __future__ import annotations

from pathlib import Path

from tools.base import ToolResult, Workspace
from tools.filesystem import read_file as _read_file
from tools.filesystem import write_file as _write_file
from tools.git import run_git
from tools.grep import grep as _grep
from tools.process import DEFAULT_OUTPUT_LIMIT_BYTES, DEFAULT_TIMEOUT_SECONDS
from tools.shell import run_shell


class Toolbox:
    """Expose shell, filesystem, grep, and Git operations in one workspace.

    A missing ``workspace_root`` creates a fresh temporary directory. Pass a
    project/task directory explicitly to operate on that workspace. Call
    ``close()`` or use this object as a context manager to clean up defaults.
    """

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        *,
        output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
    ) -> None:
        self.workspace = Workspace(workspace_root)
        self.output_limit_bytes = output_limit_bytes

    @property
    def workspace_root(self) -> Path:
        return self.workspace.root

    def shell(self, command: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> ToolResult:
        return run_shell(
            self.workspace,
            command,
            timeout,
            output_limit_bytes=self.output_limit_bytes,
        )

    def read_file(
        self, path: str, start_line: int = 1, end_line: int | None = None
    ) -> ToolResult:
        return _read_file(self.workspace, path, start_line, end_line)

    def write_file(self, path: str, content: str) -> ToolResult:
        return _write_file(self.workspace, path, content)

    def grep(self, pattern: str, path: str) -> ToolResult:
        return _grep(self.workspace, pattern, path)

    def git(self, command: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> ToolResult:
        return run_git(
            self.workspace,
            command,
            timeout,
            output_limit_bytes=self.output_limit_bytes,
        )

    def close(self) -> None:
        self.workspace.close()

    def __enter__(self) -> Toolbox:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
