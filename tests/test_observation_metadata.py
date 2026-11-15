import copy
import json
import tempfile
import unittest
from pathlib import Path

from training.agenttune.prepare_observation_metadata import SOURCE,audit_format,build_cases,metadata_cases,prepare,references
from training.agenttune.prepare_sft import read_rows


class ObservationMetadataTests(unittest.TestCase):
    def test_reviewed_json_serialization_matches_runtime_for_all_observations(self):
        audit=audit_format(references())
        self.assertEqual(audit['observation_count'],12)
        self.assertEqual(audit['runtime_serializer_identical_count'],12)
        self.assertEqual(audit['canonical_pretty_count'],12)
        self.assertEqual(audit['metadata_removed_count'],7)
        self.assertTrue(all(r['bounded_runtime_changes_only_metadata'] for r in audit['observations']))

    def test_only_metadata_changes_not_source_or_earlier_history(self):
        records=[r for r in read_rows(SOURCE) if r['arm']=='adapter' and r['condition']=='full_source']
        for record in records:
            before=copy.deepcopy(record)
            runtime,reviewed=metadata_cases(record,{'path':'shop/config.py','line_count':18})
            self.assertEqual(runtime['messages'],record['messages'])
            self.assertEqual(runtime['messages'][:-1],reviewed['messages'][:-1])
            a=json.loads(runtime['messages'][-1]['content'].split('\n',1)[1]);b=json.loads(reviewed['messages'][-1]['content'].split('\n',1)[1])
            self.assertEqual(b,{**a,'metadata':{'path':'shop/config.py','line_count':18}})
            self.assertEqual(a['stdout'],b['stdout'])
            self.assertEqual(record,before)
            with self.assertRaises(ValueError):metadata_cases(record,{'path':'other.py','line_count':18})

    def test_bundle_contains_no_gold_targets_and_uses_six_fixed_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle'
            manifest=prepare(root)
            self.assertFalse(manifest['gold_targets_bundled'])
            self.assertEqual(manifest['prediction_count'],12)
            self.assertEqual(read_rows(root/'data/cases.jsonl'),build_cases(references()))
            self.assertFalse((root/'data/approved.jsonl').exists())
            self.assertFalse((root/'evals').exists())
            self.assertFalse((root/'training/agenttune/sft_smoke.py').exists())


if __name__=='__main__':unittest.main()
