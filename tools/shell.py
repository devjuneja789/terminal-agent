"""Workspace-rooted shell execution with conservative destructive-command checks."""

from __future__ import annotations

import re
import shlex

from tools.base import ToolResult, Workspace
from tools.process import DEFAULT_OUTPUT_LIMIT_BYTES, DEFAULT_TIMEOUT_SECONDS, append_log, execute

_FORBIDDEN_COMMANDS = {
    "sudo", "su", "doas", "mkfs", "wipefs", "fdisk", "parted",
    "shutdown", "reboot", "poweroff", "halt", "shred",
}
_SHELL_OPERATORS = {";", "&&", "||", "|", "(", "{"}


def _safety_error(command: str) -> str | None:
    """Reject obvious host-destructive or cwd-escaping shell operations."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        return f"invalid shell command syntax: {exc}"

    command_start = True
    segments: list[list[str]] = []
    current_segment: list[str] = []
    for token in tokens:
        if token in _SHELL_OPERATORS:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            command_start = True
            continue
        current_segment.append(token)
        if command_start:
            executable = token.rsplit("/", 1)[-1]
            if executable in _FORBIDDEN_COMMANDS or executable.startswith("mkfs."):
                return f"command is blocked by the safety policy: {executable}"
            if executable in {"cd", "pushd", "popd"}:
                return f"{executable} is blocked; shell commands always run from the workspace root"
            command_start = False
    if current_segment:
        segments.append(current_segment)

    for segment in segments:
        executable = segment[0].rsplit("/", 1)[-1]
        args = segment[1:]
        if executable == "rm":
            if any(arg == "--recursive" or (arg.startswith("-") and not arg.startswith("--") and "r" in arg[1:]) for arg in args):
                return "recursive rm is blocked by the safety policy"
            if any(arg.startswith("/") or arg == ".." or arg.startswith("../") or arg.startswith("~") for arg in args):
                return "rm paths outside the workspace are blocked by the safety policy"
        if executable == "dd" and any(arg.startswith("of=") for arg in args):
            return "dd with an output target is blocked by the safety policy"
        if executable == "find" and "-delete" in args:
            return "find -delete is blocked by the safety policy"
        if executable in {"chmod", "chown"} and any(arg in {"-R", "--recursive"} for arg in args):
            return "recursive permission changes are blocked by the safety policy"
        if executable == "git" and len(args) >= 2:
            if args[0] == "reset" and "--hard" in args[1:]:
                return "git reset --hard is blocked by the safety policy"
            if args[0] == "clean" and any(arg.startswith("-f") for arg in args[1:]):
                return "forced git clean is blocked by the safety policy"
    if re.search(r"\b:\s*\(\s*\)\s*\{", command):
        return "fork-bomb syntax is blocked by the safety policy"
    return None


def run_shell(
    workspace: Workspace,
    command: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    *,
    output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
) -> ToolResult:
    """Run a shell command from the workspace root.

    Child processes receive only a small allowlist of environment variables;
    HOME, temp, XDG, and Git config locations are redirected into the workspace.
    """
    if not isinstance(command, str) or not command.strip():
        message = "command must be a non-empty string"
        append_log(workspace, {"tool": "shell", "command": str(command), "error": message})
        return ToolResult(success=False, error=message)

    policy_error = _safety_error(command)
    if policy_error:
        append_log(workspace, {"tool": "shell", "command": command, "error": policy_error})
        return ToolResult(success=False, error=policy_error)

    return execute(
        workspace,
        ["/bin/bash", "--noprofile", "--norc", "-c", command],
        command_for_log=command,
        timeout=timeout,
        output_limit_bytes=output_limit_bytes,
        tool_name="shell",
    )
