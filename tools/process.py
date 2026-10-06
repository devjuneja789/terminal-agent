"""Bounded subprocess execution shared by shell and Git tools."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from tools.base import ToolResult, Workspace

DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 300
DEFAULT_OUTPUT_LIMIT_BYTES = 64 * 1024


def filtered_environment(workspace: Workspace) -> dict[str, str]:
    """Build a small child environment without inheriting credentials or HOME."""
    env: dict[str, str] = {}
    for name in ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM"):
        value = os.environ.get(name)
        if value:
            env[name] = value

    temp_dir = workspace.resolve(".terminal-slm/tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    config_dir = workspace.resolve(".terminal-slm/config")
    cache_dir = workspace.resolve(".terminal-slm/cache")
    for directory in (config_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(workspace.root),
            "TMPDIR": str(temp_dir),
            "XDG_CONFIG_HOME": str(config_dir),
            "XDG_CACHE_HOME": str(cache_dir),
            "PYTHONNOUSERSITE": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CEILING_DIRECTORIES": str(workspace.root.parent),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_VALUE_0": os.devnull,
            "GIT_CONFIG_KEY_1": "core.pager",
            "GIT_CONFIG_VALUE_1": "cat",
        }
    )
    return env


def append_log(workspace: Workspace, entry: dict[str, object]) -> None:
    log_path = workspace.resolve(".terminal-slm/tool.log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if "timestamp" not in entry:
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, sort_keys=True) + "\n")


def _drain(stream: object, limit: int, store: list[bytes], clipped: list[bool]) -> None:
    # Popen's file objects expose read(); the loose type keeps this helper
    # independent of platform-specific buffered stream classes.
    read = getattr(stream, "read")
    kept = 0
    while True:
        chunk: bytes = read(8192)
        if not chunk:
            break
        remaining = limit - kept
        if remaining > 0:
            part = chunk[:remaining]
            store.append(part)
            kept += len(part)
        if len(chunk) > max(remaining, 0):
            clipped[0] = True


def execute(
    workspace: Workspace,
    argv: Sequence[str],
    *,
    command_for_log: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
    tool_name: str,
    shell: bool = False,
) -> ToolResult:
    """Run a child process with bounded output, time, cwd, and environment."""
    if not 0 < timeout <= MAX_TIMEOUT_SECONDS:
        message = f"timeout must be greater than 0 and at most {MAX_TIMEOUT_SECONDS} seconds"
        append_log(workspace, _log_entry(tool_name, command_for_log, timeout, None, message))
        return ToolResult(success=False, error=message)
    if output_limit_bytes < 0:
        message = "output limit must not be negative"
        append_log(workspace, _log_entry(tool_name, command_for_log, timeout, None, message))
        return ToolResult(success=False, error=message)

    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    stdout_parts: list[bytes] = []
    stderr_parts: list[bytes] = []
    stdout_clipped = [False]
    stderr_clipped = [False]
    exit_code: int | None = None
    error: str | None = None

    try:
        process = subprocess.Popen(
            argv,
            cwd=workspace.root,
            env=filtered_environment(workspace),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        stdout_thread = threading.Thread(
            target=_drain, args=(process.stdout, output_limit_bytes, stdout_parts, stdout_clipped), daemon=True
        )
        stderr_thread = threading.Thread(
            target=_drain, args=(process.stderr, output_limit_bytes, stderr_parts, stderr_clipped), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            exit_code = process.returncode
        stdout_thread.join()
        stderr_thread.join()
        if timed_out:
            error = f"command timed out after {timeout:g} seconds"
    except OSError as exc:
        error = f"could not start command: {exc}"
    finally:
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

    stdout = b"".join(stdout_parts).decode("utf-8", errors="replace")
    stderr = b"".join(stderr_parts).decode("utf-8", errors="replace")
    append_log(
        workspace,
        _log_entry(
            tool_name,
            command_for_log,
            timeout,
            exit_code,
            error,
            stdout_truncated=stdout_clipped[0],
            stderr_truncated=stderr_clipped[0],
        ),
    )
    return ToolResult(
        success=error is None and exit_code == 0,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        error=error,
        stdout_truncated=stdout_clipped[0],
        stderr_truncated=stderr_clipped[0],
    )


def _log_entry(
    tool: str,
    command: str,
    timeout: float,
    exit_code: int | None,
    error: str | None,
    *,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
) -> dict[str, object]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "command": command,
        "timeout_seconds": timeout,
        "exit_code": exit_code,
        "error": error,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
