"""Workspace-confined file read and write tools."""

from __future__ import annotations

from pathlib import Path

from tools.base import ToolResult, Workspace


def read_file(
    workspace: Workspace,
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> ToolResult:
    """Read an inclusive, 1-based line range from a UTF-8 workspace file."""
    try:
        if start_line < 1 or (end_line is not None and end_line < start_line):
            raise ValueError("line range must be 1-based and end_line must be >= start_line")
        target = workspace.resolve(path)
        if not target.is_file():
            raise ValueError("path is not a file")
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
        selected = lines[start_line - 1 : end_line]
        return ToolResult(
            success=True,
            stdout="".join(selected),
            metadata={"path": target.relative_to(workspace.root).as_posix(), "line_count": len(lines)},
        )
    except (OSError, UnicodeError, ValueError) as exc:
        return ToolResult(success=False, error=str(exc))


def write_file(workspace: Workspace, path: str, content: str) -> ToolResult:
    """Write UTF-8 text to a workspace-relative path, creating parent folders."""
    try:
        target = workspace.resolve(path)
        if target.exists() and target.is_dir():
            raise ValueError("path points to a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(
            success=True,
            stdout=f"Wrote {len(content)} characters to {target.relative_to(workspace.root).as_posix()}.",
            metadata={"path": target.relative_to(workspace.root).as_posix(), "characters_written": len(content)},
        )
    except (OSError, UnicodeError, ValueError) as exc:
        return ToolResult(success=False, error=str(exc))
