"""Metric calculations for benchmark task records and aggregate runs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def task_metrics(record: dict[str, Any]) -> dict[str, Any]:
    trajectory = record.get("trajectory") or {}
    events = trajectory.get("events", [])
    model_events = [event for event in events if event.get("kind") == "model"]
    tool_events = [event for event in events if event.get("kind") == "tool"]
    emitted = sum((event.get("model_output") or "").count("<tool_call>") for event in model_events)
    valid = min(emitted, len(tool_events))
    invalid = max(0, emitted - valid)
    required_tools = set(record.get("success_condition", {}).get("required_tools", []))
    selected_tools = {
        event.get("tool_name") for event in tool_events if event.get("tool_name")
    }
    selected_required = required_tools & selected_tools
    unexpected_tools = selected_tools - required_tools
    selection_denominator = len(required_tools) + len(unexpected_tools)
    tokens = record.get("token_usage")
    total_tokens = None if tokens is None else tokens.get("total_tokens")
    return {
        "model_turns": len(model_events),
        "tool_steps": sum(bool(event.get("executed")) for event in tool_events),
        "tool_calls_attempted": emitted,
        "tool_calls_schema_valid": valid,
        "tool_calls_invalid": invalid,
        "argument_schema_validity": (valid / emitted) if emitted else None,
        "invalid_tool_call_rate": (invalid / emitted) if emitted else 0.0,
        "required_tools": sorted(required_tools),
        "selected_tools": sorted(selected_tools),
        "tool_selection_accuracy": (
            len(selected_required) / selection_denominator if selection_denominator else None
        ),
        "tokens": tokens,
        "total_tokens": total_tokens,
        "latency_seconds": record.get("latency_seconds"),
    }


def aggregate_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    count = len(rows)
    metric_rows = [task_metrics(row) for row in rows]
    attempted = sum(row["tool_calls_attempted"] for row in metric_rows)
    valid = sum(row["tool_calls_schema_valid"] for row in metric_rows)
    invalid = sum(row["tool_calls_invalid"] for row in metric_rows)
    required = sum(len(row["required_tools"]) for row in metric_rows)
    selected_required = sum(
        len(set(row["required_tools"]) & set(row["selected_tools"])) for row in metric_rows
    )
    unexpected = sum(
        len(set(row["selected_tools"]) - set(row["required_tools"])) for row in metric_rows
    )
    selection_denominator = required + unexpected
    tokens = [row["total_tokens"] for row in metric_rows if row["total_tokens"] is not None]
    latencies = [row["latency_seconds"] for row in metric_rows if row["latency_seconds"] is not None]
    failures = Counter(row.get("failure_category", "unknown") for row in rows if not row.get("success"))
    return {
        "task_count": count,
        "successful_tasks": sum(bool(row.get("success")) for row in rows),
        "task_success_rate": (
            sum(bool(row.get("success")) for row in rows) / count if count else None
        ),
        "tool_selection_accuracy": (
            selected_required / selection_denominator if selection_denominator else None
        ),
        "argument_schema_validity": valid / attempted if attempted else None,
        "invalid_tool_call_rate": invalid / attempted if attempted else 0.0,
        "average_tool_steps_per_task": (
            sum(row["tool_steps"] for row in metric_rows) / count if count else None
        ),
        "average_model_turns_per_task": (
            sum(row["model_turns"] for row in metric_rows) / count if count else None
        ),
        "average_tokens_per_task": sum(tokens) / len(tokens) if tokens else None,
        "token_metric_tasks": len(tokens),
        "average_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "failure_categories": dict(sorted(failures.items())),
    }
