"""Build deterministic, human-authored reference traces from train-support tasks.

This does not call a model or train anything. Tool observations and validators
are executed against disposable snapshot copies, and every row remains excluded
from training until a maintainer reviews and approves it.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.benchmark.loader import load_benchmark
from tools import Toolbox
from training.agenttune.support_prompt import SUPPORT_SYSTEM_PROMPT


def _action(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"tool_name": name, "parameters": arguments}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _reference_actions(task_id: str, snapshot: Path) -> tuple[list[dict[str, Any]], str]:
    if task_id == "B006":
        return [
            _action("grep", {"pattern": "def resolve_route", "path": "shop/api.py"}),
            _action("read_file", {"path": "shop/api.py", "start_line": 11, "end_line": 33}),
        ], "`resolve_route` is in `shop/api.py`. It looks up the method/path key in `ROUTES`, validates item IDs with `is_positive_id`, and returns the matching handler or `None`."
    if task_id == "B016":
        return [
            _action("grep", {"pattern": "REQUEST_TIMEOUT_SECONDS", "path": "shop/config.py"}),
            _action("read_file", {"path": "shop/config.py", "start_line": 8, "end_line": 13}),
        ], "`REQUEST_TIMEOUT_SECONDS` is named by `REQUEST_TIMEOUT_ENV`, but `request_timeout()` does not read that environment variable; it returns `DEFAULT_TIMEOUT_SECONDS` instead."
    if task_id == "B021":
        content = (snapshot / "shop/config.py").read_text(encoding="utf-8")
        fixed = content.replace("return min(value, MAX_PAGE_SIZE)", "return max(1, min(value, MAX_PAGE_SIZE))")
        if fixed == content:
            raise RuntimeError("B021 reference edit did not match the snapshot")
        return [
            _action("read_file", {"path": "shop/config.py", "start_line": 1, "end_line": 30}),
            _action("write_file", {"path": "shop/config.py", "content": fixed}),
            _action("shell", {"command": "python3 -m unittest tests.test_contracts.Contracts.test_page_size_lower_bound", "timeout": 30}),
        ], "Fixed `page_size()` to clamp values to at least 1. The named test passes."
    if task_id == "B026":
        content = (snapshot / "shop/config.py").read_text(encoding="utf-8")
        fixed = content.replace(
            "return DEFAULT_TIMEOUT_SECONDS",
            "return int(os.environ.get(REQUEST_TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS))",
        )
        if fixed == content:
            raise RuntimeError("B026 reference edit did not match the snapshot")
        return [
            _action("shell", {"command": "python3 -m unittest tests.test_contracts.Contracts.test_timeout_from_environment", "timeout": 30}),
            _action("read_file", {"path": "shop/config.py", "start_line": 1, "end_line": 30}),
            _action("write_file", {"path": "shop/config.py", "content": fixed}),
            _action("shell", {"command": "python3 -m unittest tests.test_contracts.Contracts.test_timeout_from_environment", "timeout": 30}),
        ], "Updated `request_timeout()` to parse `REQUEST_TIMEOUT_ENV` with the default as its fallback. The named test passes."
    if task_id == "B041":
        content = (snapshot / "shop/config.py").read_text(encoding="utf-8")
        fixed = content.replace("DEFAULT_TIMEOUT_SECONDS = 10", "DEFAULT_TIMEOUT_SECONDS = 15")
        if fixed == content:
            raise RuntimeError("B041 reference edit did not match the snapshot")
        return [
            _action("read_file", {"path": "shop/config.py", "start_line": 1, "end_line": 30}),
            _action("write_file", {"path": "shop/config.py", "content": fixed}),
            _action("shell", {"command": "python3 -m unittest tests.test_contracts.Contracts.test_timeout_default", "timeout": 30}),
        ], "Changed `DEFAULT_TIMEOUT_SECONDS` to 15. The named test passes."
    if task_id == "B046":
        return [
            _action("shell", {"command": "python3 -m unittest tests.test_contracts.Contracts.test_timeout_from_environment", "timeout": 30}),
            _action("read_file", {"path": "shop/config.py", "start_line": 8, "end_line": 13}),
        ], "The test fails with `AssertionError: 10 != 25`: it expects the patched `REQUEST_TIMEOUT_SECONDS` value 25, but `request_timeout()` returns `DEFAULT_TIMEOUT_SECONDS` (10) and does not read the environment variable. No files were changed."
    raise ValueError(f"no reference actions are defined for {task_id}")


def _run_validator(command: tuple[str, ...], workspace: Path) -> dict[str, Any]:
    import os
    import subprocess

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(workspace),
        "TMPDIR": str(workspace / ".tmp"),
    }
    Path(env["TMPDIR"]).mkdir(exist_ok=True)
    result = subprocess.run(
        command, cwd=workspace, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=60, check=False,
    )
    return {"command": list(command), "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _task_file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    ignored = {".terminal-slm", "__pycache__", ".git"}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in ignored for part in relative.parts) or path.suffix == ".pyc":
            continue
        hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def build(output: Path, *, task_ids: tuple[str, ...] | None = None, version: int = 2) -> list[dict[str, Any]]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reference or reviewed traces: {output}")
    tasks = {task.task_id: task for task in load_benchmark("train-support")}
    selected_ids = task_ids or ("B006", "B016", "B021", "B026", "B041", "B046")
    if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids).issubset(tasks):
        raise ValueError("reference task IDs must be unique train-support IDs")
    if version < 1:
        raise ValueError("version must be positive")
    records: list[dict[str, Any]] = []
    for task_id in selected_ids:
        task = tasks[task_id]
        with tempfile.TemporaryDirectory(prefix=f"terminal-slm-gold-{task_id}-") as temp:
            workspace = Path(temp) / "workspace"
            shutil.copytree(task.snapshot_path, workspace)
            before_hashes = _task_file_hashes(workspace)
            initial_validation = _run_validator(task.validation_command, workspace)
            actions, final_answer = _reference_actions(task_id, task.snapshot_path)
            messages: list[dict[str, str]] = [
                {"role": "system", "content": SUPPORT_SYSTEM_PROMPT},
                {"role": "user", "content": task.instruction},
            ]
            invoked_tools: list[str] = []
            tool_results: list[dict[str, Any]] = []
            with Toolbox(workspace) as toolbox:
                for action in actions:
                    messages.append({"role": "assistant", "content": _json([action])})
                    name = action["tool_name"]
                    result = getattr(toolbox, name)(**action["parameters"]).as_dict()
                    invoked_tools.append(name)
                    tool_results.append({"tool": name, "arguments": action["parameters"], "result": result})
                    messages.append({
                        "role": "user",
                        "content": "Tool result (JSON):\n" + json.dumps(result, ensure_ascii=False, indent=2),
                    })
                    if result.get("error") is not None or (
                        result.get("exit_code") is None and result["success"] is not True
                    ):
                        raise RuntimeError(f"{task_id}: reference tool failed: {result}")
                finish = _action("finish", {"answer": final_answer})
                messages.append({"role": "assistant", "content": _json([finish])})

            validation = _run_validator(task.validation_command, workspace)
            allowed_codes = task.success_condition.get("validation_exit_codes", [0])
            if validation["exit_code"] not in allowed_codes:
                raise RuntimeError(f"{task_id}: reference validation failed: {validation}")
            after_hashes = _task_file_hashes(workspace)
            changed_files = sorted(
                path for path in set(before_hashes) | set(after_hashes)
                if before_hashes.get(path) != after_hashes.get(path)
            )
            changed = bool(changed_files)
            if task_id in {"B021", "B026", "B041"} and not changed:
                raise RuntimeError(f"{task_id}: expected a verified source change")
            if task_id == "B046" and changed:
                raise RuntimeError("B046 is read-only but the snapshot changed")
            if task_id in {"B021", "B026", "B041"} and initial_validation["exit_code"] == 0:
                raise RuntimeError(f"{task_id}: expected the pristine validator to fail")

            response_terms = task.success_condition.get("response_contains", [])
            missing_response_terms = [term for term in response_terms if term.casefold() not in final_answer.casefold()]
            required_tools = set(task.success_condition.get("required_tools", []))
            if missing_response_terms or not required_tools.issubset(invoked_tools):
                raise RuntimeError(
                    f"{task_id}: reference final answer/tool sequence does not satisfy task contract; "
                    f"missing_terms={missing_response_terms}, missing_tools={sorted(required_tools - set(invoked_tools))}"
                )

            records.append({
                "schema": "terminal-slm-reference-trace-v1",
                "example_id": f"G-{task_id}-v{version}",
                "source_task_id": task_id,
                "source_split": "train-support",
                "snapshot_sha256": task.snapshot_sha256,
                "task_contract_sha256": hashlib.sha256(json.dumps({
                    "task_id": task.task_id,
                    "category": task.category,
                    "split": task.split,
                    "instruction": task.instruction,
                    "expected_outcome": task.expected_outcome,
                    "validation_command": list(task.validation_command),
                    "success_condition": task.success_condition,
                }, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
                "authorship": "assistant-authored canonical reference; maintainer review required",
                "system_prompt_sha256": hashlib.sha256(SUPPORT_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "messages": messages,
                "reference_audit": {
                    "invoked_tools": invoked_tools,
                    "tool_results": tool_results,
                    "initial_validation": initial_validation,
                    "validation": validation,
                    "workspace_changed": changed,
                    "changed_files": changed_files,
                    "human_review_status": "pending",
                    "training_eligible": False,
                },
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--version", type=int, default=2)
    args = parser.parse_args()
    destination = args.output
    records = build(destination, task_ids=tuple(args.task_ids) if args.task_ids else None, version=args.version)
    print(f"Wrote {len(records)} verified train-support reference traces to {destination}")
    print("All traces require maintainer review and are marked training_eligible=false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
