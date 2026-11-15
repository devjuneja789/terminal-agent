import copy
import json
import tempfile
import unittest
from pathlib import Path

from training.agenttune.build_corrective_support import build,proposed_action_rows
from training.agenttune.prepare_observation_intervention import REPLAY
from training.agenttune.prepare_sft import read_rows


class CorrectiveSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.path=Path(cls.temp.name)/'candidates.jsonl'
        cls.records=build(cls.path)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_actual_prefixes_are_preserved_and_never_supervision_targets(self):
        archive={r['task_id']:r for r in read_rows(REPLAY) if r['arm']=='adapter'}
        count=0
        for r in self.records:
            boundary=r['context_message_count']
            expected=archive[r['source_task_id']]['agenttune_trace']['model_prompts'][r['source_model_output_index']]
            self.assertEqual(r['messages'][:boundary],expected)
            for row in proposed_action_rows(r):
                self.assertGreaterEqual(row['source_message_index'],boundary)
                self.assertFalse(row['training_eligible'])
                count+=1
        self.assertEqual(count,15)

    def test_repairs_and_expected_failure_are_verified_separately(self):
        for r in self.records:
            audit=r['reference_audit']
            self.assertFalse(r['training_eligible'])
            self.assertEqual(audit['human_review_status'],'pending')
            if r['purpose']=='repair':
                self.assertEqual(audit['validation']['exit_code'],0)
                self.assertTrue(audit['task_contract_satisfied'])
                self.assertEqual(audit['changed_files_by_correction'],['shop/config.py'])
                self.assertEqual(audit['continuation_tool_results'][-1]['result']['exit_code'],0)
            elif r['purpose']=='explanation':
                self.assertEqual(audit['validation']['exit_code'],1)
                self.assertTrue(audit['task_contract_satisfied'])
                self.assertEqual(audit['changed_files_from_pristine'],[])
                self.assertIn('10 != 25',audit['validation']['stderr'])
            else:
                self.assertFalse(audit['task_contract_satisfied'])
                self.assertEqual(audit['changed_files_by_correction'],[])
                self.assertEqual(proposed_action_rows(r),[])
                self.assertEqual(r['admission_policy'],'audit_only_excluded')

    def test_runtime_result_metadata_and_schema_are_preserved(self):
        for r in self.records:
            for m in r['messages'][r['context_message_count']:]:
                if m['role']=='user':
                    self.assertTrue(m['content'].startswith('Tool result (JSON):\n'))
                    result=json.loads(m['content'].split('\n',1)[1])
                    self.assertEqual(result['metadata'],{})
                else:
                    action=json.loads(m['content'])
                    self.assertEqual(len(action),1)
                    self.assertEqual(set(action[0]),{'tool_name','parameters'})

    def test_review_hashes_and_overwrite_guard(self):
        from training.agenttune.prepare_sft import digest
        reviews=read_rows(self.path.with_suffix('.review.jsonl'))
        self.assertEqual(len(reviews),6)
        for r,review in zip(self.records,reviews,strict=True):
            self.assertEqual(review['source_record_sha256'],digest(r))
            self.assertEqual(review['decision'],'pending')
        packet=self.path.with_suffix('.review.md').read_text()
        self.assertIn(self.records[0]['messages'][1]['content'],packet)
        self.assertIn('Last archived tool observation (context only)',packet)
        with self.assertRaises(FileExistsError):build(self.path)
        bad=copy.deepcopy(self.records[0]);bad['messages'][1]['content']='changed task'
        with self.assertRaisesRegex(ValueError,'context changed'):proposed_action_rows(bad)


if __name__=='__main__':unittest.main()
