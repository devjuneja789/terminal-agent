"""Shared workspace and result types for terminal tools."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


class ToolResult:
    """A predictable result envelope returned by every tool."""

    def __init__(
        self,
        *,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        error: str | None = None,
        stdout_truncated: bool = False,
        stderr_truncated: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.error = error
        self.stdout_truncated = stdout_truncated
        self.stderr_truncated = stderr_truncated
        self.metadata = metadata or {}

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation suitable for an agent."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "error": self.error,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "metadata": self.metadata,
        }


class Workspace:
    """A root directory that bounds file tools and is the default process cwd.

    With no path, a private temporary directory is created outside the user's
    home directory. Explicit roots may be inside a project, but may not be the
    home directory itself or an ancestor that would expose all of it.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        if root is None:
            temporary_root = Path("/tmp") if Path("/tmp").is_dir() else Path(tempfile.gettempdir())
            self._temporary = tempfile.TemporaryDirectory(prefix="terminal-slm-", dir=temporary_root)
            requested_root = Path(self._temporary.name)
        else:
            requested_root = Path(root).expanduser()
            requested_root.mkdir(parents=True, exist_ok=True)

        self.root = requested_root.resolve(strict=True)
        home = Path.home().resolve()
        if self.root == home or home.is_relative_to(self.root):
            self.close()
            raise ValueError("workspace root must not be the home directory or its ancestor")

        internal_dir = self.root / ".terminal-slm"
        if internal_dir.is_symlink():
            self.close()
            raise ValueError(".terminal-slm must not be a symlink")
        internal_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = internal_dir / "tool.log.jsonl"

    @property
    def log_path(self) -> Path:
        return self._log_path

    def resolve(self, relative_path: str | Path) -> Path:
        """Resolve a workspace-relative path and reject escapes, including symlinks."""
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError("absolute paths are not allowed; use a workspace-relative path")
        resolved = (self.root / candidate).resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise ValueError("path escapes the workspace")
        return resolved

    def close(self) -> None:
        """Remove an automatically-created temporary workspace."""
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> Workspace:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
