"""Extract held-out evaluator records from an attached HF Job log."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any


def extract(log_path: Path, output_dir: Path, *, summary_only: bool = False) -> tuple[Path, ...]:
    environment: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    chunk_sets: dict[str, dict[str, Any]] = {}
    truncated_record_lines = 0
    for line_number, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for marker, setter in (
            ("BASELINE_ENVIRONMENT=", "environment"),
            ("BASELINE_TASK_RESULT=", "record"),
            ("BASELINE_TASK_RESULT_CHUNK=", "chunk"),
            ("BASELINE_SUMMARY=", "summary"),
        ):
            position = line.find(marker)
            if position < 0:
                continue
            try:
                payload = json.loads(line[position + len(marker):])
            except json.JSONDecodeError:
                if setter == "record":
                    truncated_record_lines += 1
                    break
                raise ValueError(f"malformed {setter} marker on log line {line_number}") from None
            if setter == "environment":
                environment = payload
            elif setter == "record":
                records.append(payload)
            elif setter == "chunk":
                task_id = str(payload["task_id"])
                part = int(payload["part"])
                parts = int(payload["parts"])
                data = str(payload["data"])
                if part < 1 or parts < 1 or part > parts:
                    raise ValueError(f"invalid chunk numbering for {task_id} on log line {line_number}")
                entry = chunk_sets.setdefault(task_id, {"parts": parts, "data": {}})
                if entry["parts"] != parts:
                    raise ValueError(f"inconsistent chunk count for {task_id}")
                old_data = entry["data"].get(part)
                if old_data is not None and old_data != data:
                    raise ValueError(f"conflicting duplicate chunk for {task_id} part {part}")
                entry["data"][part] = data
            else:
                summary = payload
            break
    for task_id, entry in chunk_sets.items():
        expected_parts = set(range(1, entry["parts"] + 1))
        if set(entry["data"]) != expected_parts:
            missing = sorted(expected_parts - set(entry["data"]))
            raise ValueError(f"incomplete trajectory chunks for {task_id}; missing parts {missing}")
        encoded = "".join(entry["data"][part] for part in range(1, entry["parts"] + 1))
        record = json.loads(base64.b64decode(encoded, validate=True))
        if record.get("task_id") != task_id:
            raise ValueError(f"reassembled task ID mismatch for {task_id}")
        records.append(record)
    if environment is None or summary is None:
        raise ValueError("log is missing environment or summary markers")
    ids = environment.get("task_ids") if environment else None
    actual_ids = [row.get("task_id") for row in records]
    complete_record_set = ids == actual_ids
    if not complete_record_set and not summary_only:
        detail = f" ({truncated_record_lines} legacy task record line(s) were truncated)" if truncated_record_lines else ""
        raise ValueError(
            f"task record set is incomplete: expected {ids}, got {actual_ids}{detail}; "
            "rerun with chunked trajectory logging"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = summary.get("run_id", "unknown-run")
    jsonl_path = output_dir / f"{run_id}.jsonl"
    metadata_path = output_dir / f"{run_id}.metadata.json"
    summary_path = output_dir / f"{run_id}.summary.json"
    metadata_path.write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not complete_record_set:
        partial_path: Path | None = None
        if records:
            partial_path = output_dir / f"{run_id}.partial.jsonl"
            partial_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                encoding="utf-8",
            )
        status_path = output_dir / f"{run_id}.artifact-status.json"
        status = {
            "run_id": run_id,
            "aggregate_summary_available": True,
            "complete_task_records_available": False,
            "complete_task_records_observed": len(records),
            "available_task_ids": actual_ids,
            "missing_task_ids": [task_id for task_id in ids if task_id not in actual_ids],
            "truncated_legacy_task_record_lines": truncated_record_lines,
            "note": "Aggregate metrics were printed by the completed job, but Hugging Face truncated some long per-task trajectory lines. The partial JSONL contains only complete records; it is not the full run archive.",
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        return tuple(path for path in (partial_path, metadata_path, summary_path, status_path) if path is not None)
    jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    return jsonl_path, metadata_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-only", action="store_true",
                        help="save aggregate metadata when legacy per-task log lines were truncated")
    args = parser.parse_args()
    for path in extract(args.log, args.output_dir, summary_only=args.summary_only):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
