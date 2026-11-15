"""Deterministic support-rollout reward checks; held-out records are not accepted."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any


CHANGE_REQUIRED_CATEGORIES = frozenset({
    "simple_bug_fix",
    "failing_test_fix",
    "validation_change",
    "api_route_fix",
    "configuration_update",
})

# These task contracts check keywords, tool presence, and command outcome, but
# cannot independently establish that a factual final answer is correct.
FACTUAL_REVIEW_CATEGORIES = frozenset({
    "repository_navigation",
    "function_discovery",
    "configuration_discovery",
    "environment_variable_discovery",
    "explaining_test_failure",
})

PENALTIES = {
    "invalid_action": 0.03,
    "failed_tool_execution": 0.02,
    "repeated_action": 0.02,
    "excess_tool_step": 0.01,
    "excess_tool_step_free_allowance": 4,
    "minimum_reward": -0.25,
}


def _repeated_action_count(actions: Iterable[dict[str, Any]]) -> int:
    action_list = list(actions)
    seen: dict[str, int] = {}
    repeated = 0
    for index, action in enumerate(action_list):
        name = action.get("name")
        if not name or name == "finish":
            continue
        key = json.dumps(
            {"name": name, "arguments": action.get("arguments", {})},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if key in seen:
            previous_index = seen[key]
            edited_since = any(
                prior.get("name") == "write_file" and prior.get("schema_valid")
                for prior in action_list[previous_index + 1:index]
            )
            if not edited_since:
                repeated += 1
        seen[key] = index
    return repeated


def _contract_evidence_failures(task: Any, record: dict[str, Any]) -> list[str]:
    """Recheck strengthened tool requirements against this actual trace."""
    trace = record.get("agenttune_trace", {})
    executions = trace.get("tool_executions", [])
    invoked_tools = {
        item.get("tool")
        for item in executions
        if item.get("schema_valid", True)
        and isinstance(item.get("result"), dict)
    }
    failures = [
        f"missing_required_tool:{tool}"
        for tool in task.success_condition.get("required_tools", [])
        if tool not in invoked_tools
    ]
    if task.success_condition.get("no_task_file_changes") and record.get("support_audit", {}).get("changed_files"):
        failures.append("unexpected_task_file_changes")
    for evidence in task.success_condition.get("tool_evidence", []):
        name = evidence.get("tool")
        matched = False
        for item in executions:
            if item.get("tool") != name or not item.get("schema_valid", True):
                continue
            arguments = item.get("arguments", {})
            command = str(arguments.get("command", "")).casefold()
            arguments_match = evidence.get("arguments_match", {})
            if not isinstance(arguments_match, dict) or any(
                arguments.get(key) != value for key, value in arguments_match.items()
            ):
                continue
            result = item.get("result")
            if not isinstance(result, dict) or result.get("error") is not None:
                continue
            nested = result.get("output")
            if isinstance(nested, dict):
                result = nested
            output = "\n".join(str(result.get(key, "")) for key in ("stdout", "stderr")).casefold()
            if all(str(term).casefold() in command for term in evidence.get("command_contains", [])) and all(
                str(term).casefold() in output for term in evidence.get("result_contains", [])
            ):
                matched = True
                break
        if not matched:
            failures.append(f"missing_tool_evidence:{name}")
    return failures


def load_review_sidecar(path: str | Path, *, run_id: str) -> dict[str, dict[str, Any]]:
    """Load unique, run-scoped human dispositions from JSONL."""
    reviews: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id")
        if row.get("run_id") != run_id or not isinstance(task_id, str) or not task_id:
            raise ValueError(f"review line {line_no} has a missing or mismatched run/task ID")
        if task_id in reviews:
            raise ValueError(f"duplicate review for {task_id}")
        if row.get("disposition") not in {"accepted", "rejected"}:
            raise ValueError(f"review line {line_no} has invalid disposition")
        score = row.get("audit_reward")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError(f"review line {line_no} has invalid audit_reward")
        reviews[task_id] = row
    return reviews


def apply_human_review(
    scored: dict[str, Any],
    review: dict[str, Any],
    *,
    task_id: str,
    run_id: str,
    snapshot_sha256: str,
) -> dict[str, Any]:
    """Apply a matching signed-off review without trusting stale audit fields."""
    if review.get("task_id") != task_id or review.get("run_id") != run_id:
        raise ValueError("review identity does not match rollout")
    if review.get("snapshot_sha256") != snapshot_sha256:
        raise ValueError(f"{task_id}: review snapshot hash does not match rollout")
    disposition = review["disposition"]
    confirmed = review.get("signoff") == "confirmed"
    agrees = review.get("audit_agrees") is True
    status = disposition if confirmed else "pending"
    result = dict(scored)
    result["human_review"] = {
        "status": status,
        "disposition": disposition if confirmed else None,
        "signoff": review.get("signoff"),
        "audit_agrees": agrees,
        "factual_check": review.get("factual_check"),
        "training_eligible": False,
        "reviewed_at": review.get("reviewed_at"),
    }
    result["review_audit"] = {
        "audit_reward": review.get("audit_reward"),
        "audit_reward_status": review.get("audit_reward_status"),
        "current_reward": scored.get("reward"),
        "current_reward_status": scored.get("reward_status"),
        "audit_matches_current": (
            agrees
            and review.get("audit_reward") == scored.get("reward")
        ),
        "status_matches_current": review.get("audit_reward_status") == scored.get("reward_status"),
    }
    if confirmed:
        result["reward_status"] = f"human_review_{disposition}"
        # Human factual review can establish correctness, but only when the
        # reviewer explicitly confirms agreement with the current audit.
        if agrees and result["review_audit"]["audit_matches_current"]:
            result["reward"] = review.get("audit_reward")
        else:
            result["reward"] = None
        result["automated_task_success"] = bool(
            scored.get("automated_task_success")
            and disposition == "accepted"
            and agrees
            and result["review_audit"]["audit_matches_current"]
        )
    return result


def score_support_rollout(task: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Score only validator-backed state changes; require review for factual QA.

    A successful tool call or answer keyword alone cannot earn task reward:
    `benchmark_passed` includes final response, required tools, and the task's
    validation command. Change tasks additionally must fail validation on the
    pristine snapshot and modify at least one non-log task file.
    """
    if task.split != "train-support":
        raise ValueError("support rewards accept train-support tasks only")

    audit = record["support_audit"]
    initial = audit["initial_validation"]
    accepted_codes = task.success_condition.get("validation_exit_codes", [0])
    initial_failed = (
        not initial["timed_out"] and initial["exit_code"] not in accepted_codes
    )
    change_required = task.category in CHANGE_REQUIRED_CATEGORIES
    changed_files = list(audit.get("changed_files", []))
    change_gate_passed = (
        not change_required or (initial_failed and bool(changed_files))
    )
    contract_failures = _contract_evidence_failures(task, record)
    benchmark_passed = bool(record["success"]) and not contract_failures
    automated_task_success = benchmark_passed and change_gate_passed

    trace = record["agenttune_trace"]
    actions = trace.get("selected_actions", [])
    tool_actions = [action for action in actions if action.get("name") != "finish"]
    malformed_outputs = sum(error is not None for error in trace.get("parse_errors", []))
    schema_invalid = sum(not bool(action.get("schema_valid")) for action in tool_actions)
    protocol_violations = sum(
        adaptation in {
            "additional_action_arrays_ignored_after_first",
            "trailing_text_ignored_after_selected_action",
        }
        for adaptation in trace.get("response_adaptations", [])
    )
    invalid_actions = malformed_outputs + schema_invalid + protocol_violations
    tool_executions = trace.get("tool_executions", [])
    failed_executions = 0
    for item in tool_executions:
        if not item.get("schema_valid", True):
            # Argument validation rejected this action before dispatch. It is
            # already represented by invalid_actions, not an execution fault.
            continue
        result = item.get("result")
        if not isinstance(result, dict):
            failed_executions += 1
        elif item.get("tool") == "shell" and result.get("error") is None and isinstance(result.get("exit_code"), int):
            # The shell process completed. A nonzero status is its observation
            # (often an intentionally failing test), not an executor failure.
            continue
        elif result.get("error") is not None or result.get("success") is not True:
            failed_executions += 1
    repeated_actions = _repeated_action_count(actions)
    excess_steps = max(0, len(tool_actions) - PENALTIES["excess_tool_step_free_allowance"])
    penalties = {
        "invalid_action": round(invalid_actions * PENALTIES["invalid_action"], 4),
        "failed_tool_execution": round(failed_executions * PENALTIES["failed_tool_execution"], 4),
        "repeated_action": round(repeated_actions * PENALTIES["repeated_action"], 4),
        "excess_tool_step": round(excess_steps * PENALTIES["excess_tool_step"], 4),
    }
    penalty_total = sum(penalties.values())
    review_required = task.category in FACTUAL_REVIEW_CATEGORIES

    # The current factual-task contracts are too weak for a reliable scalar.
    # Preserve a null reward until a human checks the final answer against the
    # repository snapshot; candidate records remain ineligible for training.
    reward = None if review_required else round(
        max(PENALTIES["minimum_reward"], float(automated_task_success) - penalty_total), 4
    )
    if change_required and not initial_failed:
        reward_status = "rejected_pristine_validator_passes"
    elif change_required and not changed_files:
        reward_status = "rejected_no_file_change"
    elif review_required:
        reward_status = "pending_human_factual_review"
    else:
        reward_status = "automated_validator_scored"

    return {
        "reward": reward,
        "reward_status": reward_status,
        "benchmark_contract_passed": benchmark_passed,
        "contract_evidence_failures": contract_failures,
        "automated_task_success": automated_task_success,
        "change_required": change_required,
        "pristine_validation_failed": initial_failed,
        "changed_files": changed_files,
        "change_gate_passed": change_gate_passed,
        "factual_review_required": review_required,
        "human_review": {"status": "pending", "training_eligible": False},
        "counts": {
            "tool_actions": len(tool_actions),
            "invalid_actions": invalid_actions,
            "protocol_violations": protocol_violations,
            "failed_tool_executions": failed_executions,
            "repeated_actions": repeated_actions,
            "excess_tool_steps": excess_steps,
        },
        "penalties": penalties,
        "penalty_total": round(penalty_total, 4),
        "weights": dict(PENALTIES),
    }
