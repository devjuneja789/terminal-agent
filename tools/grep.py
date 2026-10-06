"""Regular-expression search over workspace files."""

from __future__ import annotations

import re

from tools.base import ToolResult, Workspace

DEFAULT_MATCH_LIMIT = 1000


def grep(
    workspace: Workspace,
    pattern: str,
    path: str,
    *,
    max_matches: int = DEFAULT_MATCH_LIMIT,
) -> ToolResult:
    """Search a file or recursively search a directory within the workspace."""
    try:
        if max_matches < 1:
            raise ValueError("max_matches must be at least 1")
        expression = re.compile(pattern)
        target = workspace.resolve(path)
        if target.is_file():
            files = [target]
        elif target.is_dir():
            files = sorted(candidate for candidate in target.rglob("*") if candidate.is_file())
        else:
            raise ValueError("path does not exist or is not a regular file/directory")

        matches: list[str] = []
        truncated = False
        for candidate in files:
            resolved = candidate.resolve(strict=False)
            if not resolved.is_relative_to(workspace.root):
                continue
            try:
                text = resolved.read_text(encoding="utf-8")
            except UnicodeError:
                continue
            relative = resolved.relative_to(workspace.root).as_posix()
            for number, line in enumerate(text.splitlines(), start=1):
                if expression.search(line):
                    if len(matches) >= max_matches:
                        truncated = True
                        break
                    matches.append(f"{relative}:{number}:{line}")
            if truncated:
                break
        return ToolResult(
            success=True,
            stdout="\n".join(matches),
            metadata={"matches": len(matches), "truncated": truncated},
        )
    except (OSError, re.error, ValueError) as exc:
        return ToolResult(success=False, error=str(exc))
