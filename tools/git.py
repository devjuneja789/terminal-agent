"""Git operations constrained to the active workspace."""

from __future__ import annotations

import shlex
from pathlib import Path

from tools.base import ToolResult, Workspace
from tools.process import DEFAULT_OUTPUT_LIMIT_BYTES, DEFAULT_TIMEOUT_SECONDS, append_log, execute

_ALLOWED_SUBCOMMANDS = {
    "add",
    "branch",
    "commit",
    "diff",
    "log",
    "ls-files",
    "rev-parse",
    "show",
    "status",
}


def _git_safety_error(args: list[str]) -> str | None:
    if not args:
        return "git command must include a subcommand"
    if args[0].startswith("-"):
        return "Git global options are blocked; provide a workspace subcommand"
    if args[0] not in _ALLOWED_SUBCOMMANDS:
        return f"git subcommand is not available in the workspace tool: {args[0]}"
    if any(arg in {"-C", "--git-dir", "--work-tree", "--config-env"} for arg in args):
        return "git directory overrides are blocked; Git is fixed to the workspace"
    if any(arg.startswith(("--git-dir=", "--work-tree=", "--config-env=")) for arg in args):
        return "git directory overrides are blocked; Git is fixed to the workspace"
    if any(arg in {"--global", "--system"} for arg in args):
        return "global and system Git configuration changes are blocked"

    subcommand = args[0]
    rest = args[1:]
    if subcommand in {"push", "fetch", "pull", "clone", "submodule"}:
        return f"network Git command is blocked in the workspace tool: {subcommand}"
    if subcommand == "reset" and "--hard" in rest:
        return "git reset --hard is blocked by the safety policy"
    if subcommand == "clean" and any(arg.startswith("-f") for arg in rest):
        return "forced git clean is blocked by the safety policy"
    if subcommand == "config" and any(arg in {"--global", "--system", "--file"} for arg in rest):
        return "Git configuration outside the workspace is blocked"
    if subcommand == "commit" and any(arg in {"--amend", "--no-verify"} for arg in rest):
        return "history rewriting and hook bypass are blocked by the safety policy"
    if subcommand == "branch" and any(arg in {"-D", "--delete", "-f", "--force"} for arg in rest):
        return "forced branch deletion or replacement is blocked by the safety policy"
    if subcommand == "diff" and any(arg == "--ext-diff" or arg.startswith("--output") for arg in rest):
        return "external diff commands and output paths are blocked by the safety policy"
    if subcommand == "add" and any(
        arg == "--pathspec-from-file" or arg.startswith("--pathspec-from-file=") or arg == "--pathspec-file-nul"
        for arg in rest
    ):
        return "Git pathspec files are blocked; provide workspace-relative paths directly"
    if any(arg == ".." or arg.startswith("../") or arg.startswith("/") for arg in args):
        return "absolute paths and parent-directory paths are blocked in Git arguments"
    return None


def run_git(
    workspace: Workspace,
    command: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    *,
    output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
) -> ToolResult:
    """Execute a Git subcommand with cwd and all config isolated to workspace."""
    try:
        args = shlex.split(command, posix=True)
    except ValueError as exc:
        append_log(workspace, {"tool": "git", "command": command, "error": str(exc)})
        return ToolResult(success=False, error=f"invalid git command syntax: {exc}")
    policy_error = _git_safety_error(args)
    if policy_error:
        append_log(workspace, {"tool": "git", "command": command, "error": policy_error})
        return ToolResult(success=False, error=policy_error)

    git_marker = workspace.root / ".git"
    if git_marker.exists() or git_marker.is_symlink():
        probe = execute(
            workspace,
            ["git", "rev-parse", "--absolute-git-dir"],
            command_for_log="<workspace git directory check>",
            timeout=min(timeout, 5),
            output_limit_bytes=4096,
            tool_name="git",
        )
        if probe.success:
            git_dir = Path(probe.stdout.strip()).resolve(strict=False)
            if not git_dir.is_relative_to(workspace.root):
                message = "Git metadata resolves outside the workspace"
                append_log(workspace, {"tool": "git", "command": command, "error": message})
                return ToolResult(success=False, error=message)
            if not git_dir.is_dir():
                message = "Git metadata directory is not available inside the workspace"
                append_log(workspace, {"tool": "git", "command": command, "error": message})
                return ToolResult(success=False, error=message)
    return execute(
        workspace,
        ["git", *args],
        command_for_log=command,
        timeout=timeout,
        output_limit_bytes=output_limit_bytes,
        tool_name="git",
    )
