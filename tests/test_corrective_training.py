import copy
import json
import tempfile
import unittest
from pathlib import Path

from training.agenttune.corrective_training_data import verified_rows, ROWS_SHA256
from training.agenttune.prepare_corrective_preflight import verify_export
from training.agenttune.prepare_corrective_training import prepare
from training.agenttune.prepare_corrective_replay import validate_result, prepare as prepare_replay
from training.agenttune.prepare_sft import ROOT, read_rows


class CorrectiveTrainingTests(unittest.TestCase):
    def setUp(self):
        self.refs = read_rows(ROOT/'data/corrective/support_corrective.v1.admitted.jsonl')
        self.rows, _ = verify_export()

    def test_only_admitted_continuation_targets_are_exported(self):
        self.assertEqual(verified_rows(self.refs, self.rows), self.rows)
        for row in self.rows:
            self.assertGreaterEqual(row['source_message_index'], row['context_message_count'])
        modified = copy.deepcopy(self.rows)
        modified[0]['completion'][0]['content'] = 'harness fallback'
        with self.assertRaises(ValueError): verified_rows(self.refs, modified)
        modified = copy.deepcopy(self.refs)
        modified[0]['proposed_target_message_indices'].insert(0, 2)
        with self.assertRaises(ValueError): verified_rows(modified, self.rows)

    def test_audit_only_or_unadmitted_sources_are_rejected(self):
        modified = copy.deepcopy(self.refs)
        modified[0]['purpose'] = 'audit_only_failure_report'
        with self.assertRaises(ValueError): verified_rows(modified, self.rows)
        modified = copy.deepcopy(self.refs)
        modified[0]['human_review']['decision'] = 'pending'
        with self.assertRaises(ValueError): verified_rows(modified, self.rows)

    def test_bundle_is_support_only_and_bounded_by_passed_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'bundle'
            manifest = prepare(root, 'test/corrective-adapter')
            self.assertEqual(manifest['rows_sha256'], ROWS_SHA256)
            self.assertEqual(manifest['action_target_count'], 15)
            self.assertFalse((root/'evals').exists())
            self.assertFalse((root/'data/source_records.jsonl').exists())
            self.assertFalse((root/'data/reviews.jsonl').exists())
            self.assertEqual(read_rows(root/'data/corrective_rows.jsonl'), self.rows)
            config = json.loads((root/'configs/tiny_aya_sft_smoke.json').read_text())
            self.assertEqual(config['max_steps'], 60)
            self.assertEqual(config['expected_epochs'], 4)
            self.assertEqual(config['eos_token_ids'], [3,6,261001])
            self.assertTrue(manifest['runtime_gate']['runtime_stop_gate_passed'])

    def test_old_training_result_is_not_accepted_as_corrective_training(self):
        old = json.loads((ROOT/'research/results/tiny-aya-weighted-sft/20261114T160511Z.result.json').read_text())
        with self.assertRaises((KeyError,ValueError)): validate_result(old)

    def test_replay_requires_complete_training_and_keeps_native_settings(self):
        config = json.loads((ROOT/'configs/tiny_aya_corrective_sft.json').read_text())
        config['artifact_repo'] = 'test/unit-fixture'
        labels = {k:True for k in ('prompt_equals_generation_prefix','all_prompt_labels_masked',
            'contiguous_completion_labels','target_labels_equal_input_ids','both_closing_tokens_supervised',
            'earlier_closing_tokens_masked','decoded_target_equals_reviewed_action','full_template_equals_trainer_ids')}
        # Explicit test fixture, never a model result or research artifact.
        result = {'config':config,'run_id':'unit-fixture','rows_sha256':ROWS_SHA256,
            'source_examples':[r['example_id'] for r in self.refs], 'trajectory_count':4,'action_target_count':15,
            'heldout_accessed':False,'adapter_changed':True,'adapter_reload_passed':True,
            'artifact_commit':'1'*40,'training_metrics':{'train_loss':1.0},
            'training_audit':{'global_step':60,'epoch':4,'skipped_optimizer_steps':[],
                'target_mask_checks':[{'label_audit':dict(labels)} for _ in self.rows],
                'weight_annotations':[{'expected_tool':json.loads(r['completion'][0]['content'])[0]['tool_name']} for r in self.rows]}}
        validate_result(result)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'fixture.json';source.write_text(json.dumps(result))
            prepare_replay(root/'bundle',source)
            replay=json.loads((root/'bundle/configs/tiny_aya_support_rollout.json').read_text())
            native=json.loads((ROOT/'research/results/tiny-aya-native-stop-replay/20261114T220329Z.metadata.json').read_text())['config']
            replay['config_id']=native['config_id']
            for key in ('adapter_repo','adapter_revision','adapter_subfolder','source_training_run_id'):
                replay['replay'][key]=native['replay'][key]
            self.assertEqual(replay,native)
            self.assertFalse((root/'bundle/data').exists())
            self.assertFalse((root/'bundle/evals/benchmark/heldout_eval.jsonl').exists())
        broken=copy.deepcopy(result);broken['training_audit']['global_step']=59
        with self.assertRaises(ValueError):validate_result(broken)
        broken=copy.deepcopy(result);broken['training_audit']['target_mask_checks'][0]['label_audit']={}
        with self.assertRaises(ValueError):validate_result(broken)


if __name__ == '__main__': unittest.main()
