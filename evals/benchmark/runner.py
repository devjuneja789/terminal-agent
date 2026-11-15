"""Execute one task in a fresh disposable repository workspace."""

from __future__ import annotations

import math
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.loop import AgentResult
from evals.benchmark.loader import BenchmarkTask, snapshot_digest


AgentRun = Callable[[str, Path], AgentResult | str]


@dataclass(frozen=True)
class BenchmarkRun:
    task_id: str
    workspace: Path
    final_response: str
    validation_exit_code: int
    validation_stdout: str
    validation_stderr: str
    passed: bool
    trajectory: dict | None = None
    agent_error: str | None = None
    agent_stop_reason: str | None = None
    validation_timed_out: bool = False
    criterion_failures: tuple[str, ...] = ()
    observed_tools: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    validation_seconds: float = 0.0
    changed_files: tuple[str, ...] = ()


def _task_file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    ignored = {".terminal-slm", "__pycache__", ".git"}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.suffix == ".pyc" or any(part in ignored for part in relative.parts):
            continue
        if path.is_symlink():
            hashes[relative.as_posix()] = "symlink:" + os.readlink(path)
        elif path.is_file():
            hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def run_task(
    task: BenchmarkTask,
    agent: AgentRun,
    *,
    workspace_parent: Path | None = None,
    validation_timeout: float = 120.0,
) -> BenchmarkRun:
    """Copy the pinned snapshot, run the agent, then validate only that copy.

    The caller owns cleanup of the returned workspace (usually by placing the
    parent in a TemporaryDirectory). No task run writes into the source snapshot.
    """
    if not math.isfinite(validation_timeout) or validation_timeout <= 0:
        raise ValueError("validation_timeout must be finite and positive")
    if snapshot_digest(task.snapshot_path) != task.snapshot_sha256:
        raise ValueError(f"{task.task_id}: repository snapshot changed after task loading")
    parent = str(workspace_parent) if workspace_parent is not None else None
    workspace = Path(tempfile.mkdtemp(prefix=f"terminal-slm-{task.task_id}-", dir=parent))
    shutil.copytree(task.snapshot_path, workspace, dirs_exist_ok=True)
    initial_hashes = _task_file_hashes(workspace)
    run_started = time.monotonic()
    agent_error: str | None = None
    try:
        agent_result = agent(task.instruction, workspace)
    except Exception as exc:
        agent_result = None
        agent_error = f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
    agent_stop_reason: str | None = None
    if agent_result is None:
        final_response = ""
        observed_tools = set()
        agent_completed = False
        trajectory = None
    elif isinstance(agent_result, str):
        final_response = agent_result
        observed_tools: set[str] = set()
        agent_completed = True
        trajectory = None
    else:
        final_response = agent_result.final_text or ""
        agent_completed = agent_result.completed
        agent_stop_reason = agent_result.stop_reason.value
        trajectory = agent_result.trajectory.to_dict()
        observed_tools = {
            event.tool_name
            for event in agent_result.trajectory.events
            if event.kind == "tool" and event.executed and event.tool_name
        }
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(workspace),
        "TMPDIR": str(workspace / ".tmp"),
    }
    Path(env["TMPDIR"]).mkdir(exist_ok=True)
    validation_started = time.monotonic()
    validation_timed_out = False
    try:
        validation = subprocess.run(
            task.validation_command,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=validation_timeout,
            check=False,
        )
        validation_exit_code = validation.returncode
        validation_stdout = validation.stdout
        validation_stderr = validation.stderr
    except subprocess.TimeoutExpired as exc:
        validation_timed_out = True
        validation_exit_code = 124
        validation_stdout = _as_text(exc.stdout)
        validation_stderr = _as_text(exc.stderr)
    validation_seconds = time.monotonic() - validation_started
    final_hashes = _task_file_hashes(workspace)
    changed_files = tuple(sorted(
        path for path in set(initial_hashes) | set(final_hashes)
        if initial_hashes.get(path) != final_hashes.get(path)
    ))
    condition = task.success_condition
    response = final_response.casefold()
    required = [str(item).casefold() for item in condition.get("response_contains", [])]
    response_missing = [term for term in required if term not in response]
    response_ok = not response_missing
    required_tools = set(condition.get("required_tools", []))
    tools_ok = required_tools.issubset(observed_tools)
    validation_output = (validation_stdout + "\n" + validation_stderr).casefold()
    validation_missing = [
        str(term)
        for term in condition.get("validation_output_contains", [])
        if str(term).casefold() not in validation_output
    ]
    trajectory_events = trajectory.get("events", []) if isinstance(trajectory, dict) else []
    tool_output = "\n".join(
        str(event.get("tool_result", {}).get(key, ""))
        for event in trajectory_events
        if event.get("kind") == "tool" and isinstance(event.get("tool_result"), dict)
        for key in ("stdout", "stderr")
    ).casefold()
    tool_output_missing = [
        str(term)
        for term in condition.get("tool_output_contains", [])
        if str(term).casefold() not in tool_output
    ]
    missing_tool_evidence: list[str] = []
    for evidence in condition.get("tool_evidence", []):
        tool_name = evidence.get("tool")
        command_terms = evidence.get("command_contains", [])
        result_terms = evidence.get("result_contains", [])
        arguments_match = evidence.get("arguments_match", {})
        matched = False
        for event in trajectory_events:
            if event.get("kind") != "tool" or event.get("tool_name") != tool_name or event.get("executed") is False:
                continue
            arguments = event.get("tool_arguments", {})
            command = str(arguments.get("command", "")).casefold()
            if not isinstance(arguments_match, dict) or any(
                arguments.get(key) != value for key, value in arguments_match.items()
            ):
                continue
            result = event.get("tool_result")
            if isinstance(result, dict) and isinstance(result.get("output"), dict):
                result = result["output"]
            output = "\n".join(
                str(result.get(key, "")) for key in ("stdout", "stderr")
            ).casefold() if isinstance(result, dict) else str(result).casefold()
            if all(str(term).casefold() in command for term in command_terms) and all(
                str(term).casefold() in output for term in result_terms
            ):
                matched = True
                break
        if not matched:
            missing_tool_evidence.append(str(tool_name))
    success_exit_codes = condition.get("validation_exit_codes", [0])
    failures = [f"missing_response_term:{term}" for term in response_missing]
    failures.extend(f"missing_required_tool:{name}" for name in sorted(required_tools - observed_tools))
    failures.extend(f"missing_validation_output:{term}" for term in validation_missing)
    failures.extend(f"missing_tool_output:{term}" for term in tool_output_missing)
    failures.extend(f"missing_tool_evidence:{name}" for name in missing_tool_evidence)
    if validation_exit_code not in success_exit_codes:
        failures.append("validation_timeout" if validation_timed_out else "validation_command_failed")
    if not agent_completed:
        failures.append("agent_incomplete")
    if condition.get("no_task_file_changes") and changed_files:
        failures.append("unexpected_task_file_changes")
    passed = not failures
    return BenchmarkRun(
        task_id=task.task_id,
        workspace=workspace,
        final_response=final_response,
        validation_exit_code=validation_exit_code,
        validation_stdout=validation_stdout,
        validation_stderr=validation_stderr,
        passed=passed,
        trajectory=trajectory,
        agent_error=agent_error,
        agent_stop_reason=agent_stop_reason,
        validation_timed_out=validation_timed_out,
        criterion_failures=tuple(failures),
        observed_tools=tuple(sorted(observed_tools)),
        elapsed_seconds=time.monotonic() - run_started,
        validation_seconds=validation_seconds,
        changed_files=changed_files,
    )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
