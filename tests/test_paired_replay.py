import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.extract_paired_replay import extract
from training.agenttune.prepare_replay_bundle import TASK_IDS, prepare
from training.agenttune.paired_replay import emit_record, verify_tokenizers
from training.agenttune import hf_gpu_smoke
from tools.definitions import get_tool_schemas


class PairedReplayTests(unittest.TestCase):
    def test_template_storage_difference_is_accepted_but_semantic_difference_rejected(self):
        base = SimpleNamespace(chat_template={'default': 'same'}, get_chat_template=lambda: 'same',
            get_vocab=lambda: {'a': 1}, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        saved = SimpleNamespace(chat_template='same', get_chat_template=lambda: 'same',
            get_vocab=lambda: {'a': 1}, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        self.assertTrue(verify_tokenizers(base, saved)['vocabulary_equal'])
        saved.get_chat_template = lambda: 'changed'
        with self.assertRaisesRegex(RuntimeError, 'effective template'):
            verify_tokenizers(base, saved)
        saved.get_chat_template = lambda: 'same'
        saved.eos_token_id = 9
        with self.assertRaisesRegex(RuntimeError, 'special token'):
            verify_tokenizers(base, saved)
    def test_bundle_restricts_tasks_and_pins_adapter_and_common_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'bundle'
            prepare(root)
            tasks = [json.loads(line) for line in (root/'evals/benchmark/train_support.jsonl').read_text().splitlines()]
            self.assertEqual(tuple(t['task_id'] for t in tasks), TASK_IDS)
            self.assertFalse((root/'evals/benchmark/heldout_eval.jsonl').exists())
            self.assertFalse((root/'configs/tiny_aya_eval.json').exists())
            self.assertFalse((root/'data').exists())
            config = json.loads((root/'configs/tiny_aya_support_rollout.json').read_text())
            self.assertEqual(config['replay']['arms'], ['base', 'adapter'])
            self.assertEqual(len(config['replay']['adapter_revision']), 40)
            self.assertEqual(config['replay']['quantization']['bnb_4bit_compute_dtype'], 'float16')

    def test_v5_observations_match_json_training_format(self):
        policy = hf_gpu_smoke.TinyAyaPolicy(None, None, None, get_tool_schemas())
        state = SimpleNamespace(events=[
            SimpleNamespace(kind='observation', payload={'text': 'Task'}),
            SimpleNamespace(kind='observation', payload={'text': "Tool result:\n{'success': True, 'stdout': 'evidence'}"})])
        policy.assistant_turns = ['[]']
        policy.raw_outputs = ['[]']
        with patch.dict(hf_gpu_smoke.EVAL_CONFIG, {'prompt_variant': 'support-v5-repeat-finish-retry'}):
            message = policy._messages(state)[-1]['content']
        self.assertTrue(message.startswith('Tool result (JSON):\n'))
        self.assertEqual(json.loads(message.split('\n', 1)[1])['success'], True)

    def _log(self):
        ids = [f'{arm}:{task}' for arm in ['base', 'adapter'] for task in TASK_IDS]
        metrics = {arm: {'successful_tasks': 0, 'task_count': 4, 'invalid_tool_call_rate': 1,
            'average_tool_steps_per_task': 0, 'average_latency_seconds': 1} for arm in ['base', 'adapter']}
        stream = io.StringIO()
        stream.write('PAIRED_ENVIRONMENT='+json.dumps({'record_ids': ids})+'\n')
        with contextlib.redirect_stdout(stream):
            for key in ids:
                arm, task = key.split(':')
                emit_record({'record_id': key, 'arm': arm, 'task_id': task,
                    'split': 'train-support', 'success': False, 'failure_category': 'invalid_tool_call',
                    'trajectory': 'x'*30000})
        stream.write('PAIRED_SUMMARY='+json.dumps({'run_id': 'test-run', 'record_ids': ids,
            'metrics': metrics, 'interpretation': 'training replay'})+'\n')
        return stream.getvalue()

    def test_lossless_chunks_keep_base_adapter_pairs_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp)/'run.log'
            log.write_text(self._log())
            rows = extract(log, Path(temp)/'results')
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(len(row['trajectory']) == 30000 for row in rows))

    def test_incomplete_trajectories_cannot_produce_complete_results(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp)/'run.log'
            lines = self._log().splitlines()
            lines.pop(1)
            log.write_text('\n'.join(lines))
            with self.assertRaisesRegex(ValueError, 'incomplete trajectory'):
                extract(log, Path(temp)/'results')


if __name__ == '__main__':
    unittest.main()
