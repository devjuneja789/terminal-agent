"""Command-line entry point for the workspace-scoped Terminal-SLM agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, TextIO

from agent.loop import AgentLoop
from agent.state import AgentConfig, StopReason
from model.client import ModelClient
from model.local import LocalOpenAICompatibleClient, LocalTransformersClient
from tools import Toolbox
from tools.executor import ToolExecutor


EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_STEP_LIMIT = 3
EXIT_EXECUTION_LIMIT = 4
EXIT_AGENT_ERROR = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="term-agent",
        description="Run a bounded terminal agent inside an explicit or temporary workspace.",
    )
    parser.add_argument("task", nargs="*", help="task to perform; omit to read from stdin or prompt")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-path",
        default="Qwen/Qwen3-1.7B",
        help="local Transformers checkpoint path or Hugging Face identifier (default: %(default)s)",
    )
    model_group.add_argument(
        "--model-endpoint",
        help="loopback OpenAI-compatible chat-completions URL for a local model server",
    )
    parser.add_argument("--model-name", default="Qwen3-1.7B", help="model name sent to a local endpoint")
    parser.add_argument("--workspace", type=Path, help="explicit project/task directory; default is temporary")
    parser.add_argument("--max-steps", type=_positive_int, default=16, help="maximum executed tool calls")
    return parser


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _task_from_input(task_parts: Sequence[str], stdin: TextIO, stdout: TextIO) -> str:
    if task_parts:
        return " ".join(task_parts).strip()
    if not stdin.isatty():
        return stdin.read().strip()
    print("Task: ", end="", file=stdout, flush=True)
    return stdin.readline().strip()


def _make_client(args: argparse.Namespace) -> ModelClient:
    if args.model_endpoint:
        return LocalOpenAICompatibleClient(args.model_endpoint, model=args.model_name)
    return LocalTransformersClient(args.model_path)


def _print_trajectory(result: object, stdout: TextIO) -> None:
    trajectory = result.trajectory  # type: ignore[attr-defined]
    for event in trajectory.events:
        if event.kind == "model":
            if event.error:
                print(f"[model] error: {event.error}", file=stdout)
            else:
                print("[model] response received", file=stdout)
            continue

        args = ", ".join(f"{key}={value!r}" for key, value in (event.tool_arguments or {}).items())
        if not event.executed:
            print(f"[tool] {event.tool_name}({args}) skipped: {event.error or 'not executed'}", file=stdout)
            continue
        outcome = event.tool_result or {}
        status = "ok" if outcome.get("success") else "failed"
        print(f"[tool] {event.tool_name}({args}) -> {status}", file=stdout)
        for field in ("stdout", "stderr", "error"):
            content = outcome.get(field)
            if content:
                for line in str(content).splitlines():
                    print(f"  {line}", file=stdout)


def main(
    argv: Sequence[str] | None = None,
    *,
    model_client: ModelClient | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the CLI; injectable streams/client keep command behavior testable."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    task = _task_from_input(args.task, stdin, stdout)
    if not task:
        print("error: provide a task argument or task on stdin", file=stderr)
        return EXIT_USAGE_ERROR

    try:
        toolbox = Toolbox(args.workspace)
    except (OSError, ValueError) as exc:
        print(f"error: cannot create workspace: {exc}", file=stderr)
        return EXIT_USAGE_ERROR

    try:
        print(f"Workspace: {toolbox.workspace_root}", file=stdout)
        client = model_client or _make_client(args)
        loop = AgentLoop(
            client,
            ToolExecutor(toolbox),
            config=AgentConfig(max_tool_calls=args.max_steps),
        )
        result = loop.run(task)
        _print_trajectory(result, stdout)
        if result.completed:
            print("\nFinal response:", file=stdout)
            print(result.final_text or "", file=stdout)
            return EXIT_OK

        print(f"\nAgent stopped: {result.stop_reason.value}", file=stderr)
        if result.stop_reason == StopReason.MAX_TOOL_CALLS:
            return EXIT_STEP_LIMIT
        if result.stop_reason in {StopReason.EXECUTION_TIMEOUT, StopReason.CONVERSATION_LIMIT}:
            return EXIT_EXECUTION_LIMIT
        if result.stop_reason == StopReason.MODEL_ERROR:
            return EXIT_RUNTIME_ERROR
        return EXIT_AGENT_ERROR
    except Exception as exc:
        print(f"error: agent failed: {exc}", file=stderr)
        return EXIT_RUNTIME_ERROR
    finally:
        toolbox.close()


if __name__ == "__main__":
    raise SystemExit(main())
