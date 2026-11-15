import base64
import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_sft_training import extract
from training.agenttune.prepare_sft import ROOT, action_rows, digest, read_rows
from training.agenttune.prepare_sft_bundle import prepare as prepare_training
from training.agenttune.prepare_replay_bundle import prepare as prepare_replay, validate_memorization_result


class MemorizationTests(unittest.TestCase):
    def setUp(self):
        self.refs = read_rows(ROOT/'data/sft/gold_support.approved.jsonl')
        self.profile = json.loads((ROOT/'configs/tiny_aya_memorization.json').read_text())
        # Synthetic metadata used only to test validation, never research results.
        self.result = {'config': {**self.profile, 'artifact_repo': 'owner/test-adapter'},
            'run_id': 'unit-test', 'artifact_commit': '0'*40,
            'source_examples': [r['example_id'] for r in self.refs],
            'rows_sha256': digest(action_rows(self.refs)),
            'training_audit': {'global_step': 64, 'epoch': 4, 'skipped_optimizer_steps': [],
                              'log_history': [{'step': i, 'loss': 0.5} for i in range(1,65)]},
            'training_metrics': {'train_loss': 0.5}, 'adapter_changed': True, 'adapter_reload_passed': True}

    def test_training_bundle_freezes_four_epochs_and_excludes_benchmark(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'train'
            prepare_training(root, 'owner/test-adapter', ROOT/'configs/tiny_aya_memorization.json')
            profile = json.loads((root/'configs/tiny_aya_sft_smoke.json').read_text())
            self.assertEqual((profile['max_steps'], profile['expected_epochs']), (64,4))
            self.assertEqual(len(read_rows(root/'data/approved.jsonl')),4)
            self.assertFalse((root/'evals').exists())
            self.assertEqual(json.loads((ROOT/'configs/tiny_aya_sft_smoke.json').read_text())['max_steps'],8)

    def test_incomplete_skipped_or_different_training_cannot_trigger_replay(self):
        validate_memorization_result(self.result, self.refs)
        for key, value in [('global_step', 63), ('epoch', 3.9), ('skipped_optimizer_steps', [10])]:
            result = copy.deepcopy(self.result)
            result['training_audit'][key] = value
            with self.assertRaises(ValueError): validate_memorization_result(result, self.refs)
        result = copy.deepcopy(self.result)
        result['rows_sha256'] = 'changed'
        with self.assertRaises(ValueError): validate_memorization_result(result, self.refs)

    def test_replay_changes_adapter_identity_but_preserves_inference_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'result.json'
            path.write_text(json.dumps(self.result))
            base, new = Path(temp)/'base', Path(temp)/'new'
            prepare_replay(base)
            prepare_replay(new, path)
            a = json.loads((base/'configs/tiny_aya_support_rollout.json').read_text())
            b = json.loads((new/'configs/tiny_aya_support_rollout.json').read_text())
            for key in ('generation','prompt','agent','hashes','runtime'):
                self.assertEqual(a[key],b[key])
            self.assertEqual(b['replay']['adapter_revision'], '0'*40)
            self.assertEqual(b['replay']['adapter_subfolder'], 'unit-test/adapter')
            self.assertFalse((new/'data').exists())

    def test_training_history_chunks_are_lossless_and_missing_parts_rejected(self):
        encoded = base64.b64encode(json.dumps(self.result).encode()).decode()
        chunks = [encoded[i:i+1000] for i in range(0,len(encoded),1000)]
        lines = ['SFT_TRAINING_RESULT_CHUNK='+json.dumps({'part': i, 'parts':len(chunks),'data': chunk})
                 for i,chunk in enumerate(chunks,1)]
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp)/'run.log'
            log.write_text('\n'.join(lines))
            path = extract(log,Path(temp)/'out')
            self.assertEqual(json.loads(path.read_text()),self.result)
            self.assertEqual(len(read_rows(Path(temp)/'out/unit-test.training-history.jsonl')),64)
            log.write_text('\n'.join(lines[1:]))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                extract(log,Path(temp)/'bad')


if __name__ == '__main__': unittest.main()
