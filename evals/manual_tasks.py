"""Small, disposable Phase A end-to-end smoke tasks (not the formal benchmark)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.loop import AgentResult


@dataclass(frozen=True)
class ManualSmokeTask:
    task_id: str
    request: str
    verify: Callable[[AgentResult, str], tuple[bool, str]]


def _successful_tool(result: AgentResult, name: str) -> list[dict]:
    return [
        event.tool_result
        for event in result.trajectory.events
        if event.kind == "tool"
        and event.tool_name == name
        and event.executed
        and event.tool_result is not None
        and event.tool_result.get("success")
    ]


def _find_function(result: AgentResult, _: str) -> tuple[bool, str]:
    results = _successful_tool(result, "grep")
    found = any(
        "validate_tool_arguments" in str(item.get("stdout", ""))
        and "tools/definitions.py" in str(item.get("stdout", ""))
        for item in results
    )
    return found, "grep must locate validate_tool_arguments in tools/definitions.py"


def _find_environment_variable(result: AgentResult, _: str) -> tuple[bool, str]:
    results = _successful_tool(result, "grep")
    found = any("LANG" in str(item.get("stdout", "")) for item in results)
    answer = (result.final_text or "").upper()
    return found and "LANG" in answer, "grep must find LANG in tools/process.py and the final answer must name it"


def _read_prompt(result: AgentResult, _: str) -> tuple[bool, str]:
    results = _successful_tool(result, "read_file")
    found = any(
        "Qwen3's native format" in str(item.get("stdout", ""))
        for item in results
    )
    return found, "read_file must read agent/prompts.py and observe its native-format instruction"


def _run_test(result: AgentResult, _: str) -> tuple[bool, str]:
    results = _successful_tool(result, "shell")
    # unittest writes its progress dots and summary to stderr by default.
    found = any(
        "OK" in (str(item.get("stdout", "")) + str(item.get("stderr", "")))
        for item in results
    )
    return found, "shell must run the named unittest and return OK"


def _make_safe_change(result: AgentResult, workspace: str) -> tuple[bool, str]:
    expected = "Terminal-SLM manual smoke task completed.\n"
    path = f"{workspace}/research/manual_smoke_marker.txt"
    try:
        with open(path, encoding="utf-8") as marker:
            content = marker.read()
    except OSError:
        content = None
    wrote_file = any(
        event.kind == "tool"
        and event.tool_name == "write_file"
        and event.executed
        and event.tool_result is not None
        and event.tool_result.get("success")
        for event in result.trajectory.events
    )
    # The requested content is one line; a trailing newline is optional.
    return wrote_file and content in {expected, expected.rstrip("\n")}, (
        "write_file must create the exact marker in the disposable workspace"
    )


MANUAL_SMOKE_TASKS = (
    ManualSmokeTask(
        "find_function_definition",
        "Find the definition of validate_tool_arguments. Use a tool to locate it, then report its file and purpose.",
        _find_function,
    ),
    ManualSmokeTask(
        "find_environment_variable",
        "Find the environment variable copied into subprocesses to preserve their language/locale. Use a tool, then name the variable and file.",
        _find_environment_variable,
    ),
    ManualSmokeTask(
        "read_file",
        "Read agent/prompts.py and explain which tool-call format the agent is instructed to use.",
        _read_prompt,
    ),
    ManualSmokeTask(
        "run_test",
        "Run this exact test with the shell tool: python3 -m unittest tests.test_agent.NativeToolCallParserTests.test_valid_native_tool_call. Report whether it passed.",
        _run_test,
    ),
    ManualSmokeTask(
        "trivial_safe_change",
        "Using write_file, create research/manual_smoke_marker.txt with exactly this one line: Terminal-SLM manual smoke task completed.",
        _make_safe_change,
    ),
)
