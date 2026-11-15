from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from training.agenttune.prepare_sft import ROOT, action_rows, read_rows, verified_references
from training.agenttune.prepare_sft_bundle import prepare


class SFTPreparationTests(unittest.TestCase):
    def setUp(self):
        root = ROOT/'data/sft'
        self.raw = read_rows(root/'gold_support.jsonl')
        self.reviews = read_rows(root/'gold_support.review.jsonl')
        self.approved = read_rows(root/'gold_support.approved.jsonl')

    def test_changed_approved_contents_are_rejected(self):
        forged = copy.deepcopy(self.approved)
        forged[0]['messages'][-1]['content'] = 'harness fallback'
        with self.assertRaisesRegex(ValueError, 'differs'):
            verified_references(self.raw, self.reviews, forged)

    def test_rejected_review_cannot_be_promoted(self):
        reviews = copy.deepcopy(self.reviews)
        for row in reviews:
            if row['example_id'] == self.approved[0]['example_id']:
                row['decision'] = 'rejected'
        with self.assertRaisesRegex(ValueError, 'approval'):
            verified_references(self.raw, reviews, self.approved)

    def test_each_action_is_a_completion_with_observations_only_in_prompt(self):
        refs = verified_references(self.raw, self.reviews, self.approved)
        rows = action_rows(refs)
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertEqual(len(row['completion']), 1)
            self.assertEqual(row['completion'][0]['role'], 'assistant')
            self.assertEqual(row['prompt'][-1]['role'], 'user')
        # The intentionally failed B046 test remains in context, never a target.
        evidence = [r for r in rows if r['source_task_id'] == 'B046']
        self.assertTrue(any('AssertionError: 10 != 25' in m['content'] for r in evidence for m in r['prompt']))

    def test_bundle_has_only_four_approved_examples_and_no_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/'bundle'
            manifest = prepare(destination, 'owner/new-private-smoke')
            self.assertEqual(len(manifest['source_example_ids']), 4)
            self.assertFalse((destination/'evals').exists())
            self.assertEqual(len(read_rows(destination/'data/source_records.jsonl')), 4)
            self.assertTrue((destination/'training/agenttune/run_sft_job.sh').is_file())


if __name__ == '__main__':
    unittest.main()
