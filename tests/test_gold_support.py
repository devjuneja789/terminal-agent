from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path

from evals.benchmark import load_benchmark


class GoldSupportSetTests(unittest.TestCase):
    def test_owner_reviews_bind_original_records_and_exclude_rejected_versions(self) -> None:
        root = Path(__file__).resolve().parents[1] / "data/sft"
        def read(name):
            return [json.loads(line) for line in (root / name).read_text().splitlines() if line.strip()]
        originals = {row['example_id']: row for row in read('gold_support.jsonl')}
        for review in read('gold_support.review.jsonl'):
            source = originals[review['example_id']]
            digest = hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
            self.assertEqual(review['source_record_sha256'], digest)
        approved = read('gold_support.approved.jsonl')
        self.assertEqual({row['example_id'] for row in approved}, {'G-B021-v1', 'G-B026-v1', 'G-B041-v1', 'G-B046-v1'})
        self.assertTrue(all(row['reference_audit']['training_eligible'] for row in approved))
        replacements = {row['example_id']: row for row in read('gold_support.regenerated-v2.jsonl')}
        self.assertEqual(set(replacements), {'G-B006-v2', 'G-B016-v2'})
        b006 = replacements['G-B006-v2']['reference_audit']['tool_results'][1]['arguments']
        self.assertEqual((b006['start_line'], b006['end_line']), (11, 33))
        b016 = replacements['G-B016-v2']['reference_audit']['tool_results'][0]['arguments']
        self.assertEqual(b016['pattern'], 'REQUEST_TIMEOUT_SECONDS')
        self.assertTrue(all(not row['reference_audit']['training_eligible'] for row in replacements.values()))

    def test_gold_records_are_train_support_only_and_not_yet_training_eligible(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data/sft/gold_support.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        support = {task.task_id: task for task in load_benchmark("train-support")}
        self.assertEqual(
            {record["source_task_id"] for record in records},
            {"B006", "B016", "B021", "B026", "B041", "B046"},
        )
        for record in records:
            task_id = record["source_task_id"]
            self.assertIn(task_id, support)
            self.assertEqual(record["source_split"], "train-support")
            self.assertEqual(record["snapshot_sha256"], support[task_id].snapshot_sha256)
            self.assertFalse(record["reference_audit"]["training_eligible"])
            self.assertEqual(record["reference_audit"]["human_review_status"], "pending")

    def test_reference_edits_and_expected_failure_have_verified_outcomes(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data/sft/gold_support.jsonl"
        records = {row["source_task_id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
        for task_id in ("B021", "B026", "B041"):
            audit = records[task_id]["reference_audit"]
            self.assertTrue(audit["workspace_changed"])
            self.assertEqual(audit["validation"]["exit_code"], 0)
            self.assertIn("write_file", audit["invoked_tools"])
        failure = records["B046"]["reference_audit"]
        self.assertFalse(failure["workspace_changed"])
        self.assertEqual(failure["validation"]["exit_code"], 1)
        self.assertIn("AssertionError: 10 != 25", failure["validation"]["stderr"])


if __name__ == "__main__":
    unittest.main()
