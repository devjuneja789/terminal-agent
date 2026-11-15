import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evals.auditkit_report import recorded_contract_failures
from scripts.extract_paired_replay import extract
from scripts.build_submission_demo import build
from training.agenttune.paired_replay import emit_record
from training.agenttune.prepare_final_heldout import HELDOUT_IDS, prepare

ROOT = Path(__file__).resolve().parents[1]


class FinalSubmissionTests(unittest.TestCase):
    def test_report_scripts_run_directly_from_the_checkout(self):
        for script in ('audit_final_heldout.py','write_submission_results.py'):
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--help'],
                cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_walkthrough_refuses_incomplete_or_nonfinal_records(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'records.jsonl'
            source.write_text('')
            with self.assertRaisesRegex(ValueError,'20 real final'):
                build(source,Path(temp)/'demo.html')
            self.assertFalse((Path(temp)/'demo.html').exists())

    def test_missing_freeze_prevents_staging_heldout_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)/'bundle'
            with self.assertRaises(FileNotFoundError):
                prepare(destination, Path(temp)/'missing.json')
            self.assertFalse(destination.exists())

    def test_frozen_candidate_stages_only_heldout_with_same_native_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'training.json'
            source.write_text('{}')
            candidate = {'freeze_id': 'terminal-slm-submission-candidate-v1',
                'heldout_used_for_selection': False, 'further_training_planned': False,
                'selected_training_result': str(source),
                'selected_training_result_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'model_id': 'CohereLabs/tiny-aya-global',
                'model_revision': 'af89d219b53ed9b13b8a4645f9c8028973510324',
                'adapter_repo': 'fixture/adapter', 'adapter_revision': 'a'*40,
                'adapter_subfolder': 'fixture/adapter'}
            frozen = root/'freeze.json'
            frozen.write_text(json.dumps(candidate))
            prepare(root/'bundle', frozen)
            config = json.loads((root/'bundle/configs/final_heldout.json').read_text())
            self.assertEqual(config['paired']['task_ids'], HELDOUT_IDS)
            self.assertEqual(config['paired']['eos_token_ids'], [3, 6, 261001])
            self.assertEqual(config['paired']['arms'], ['base', 'adapter'])
            self.assertFalse((root/'bundle/data').exists())
            self.assertFalse((root/'bundle/evals/benchmark/train_support.jsonl').exists())
            self.assertEqual((root/'bundle/evals/benchmark/heldout_eval.jsonl').read_bytes(),
                (ROOT/'evals/benchmark/heldout_eval.jsonl').read_bytes())

    def test_heldout_extractor_requires_all_twenty_pairs_and_correct_split(self):
        ids = [f'{arm}:{task}' for arm in ['base', 'adapter'] for task in HELDOUT_IDS]
        metrics = {arm: {'successful_tasks': 0, 'task_count': 10,
            'invalid_tool_call_rate': 0, 'average_tool_steps_per_task': 0,
            'average_latency_seconds': 0} for arm in ['base', 'adapter']}
        stream = io.StringIO()
        stream.write('PAIRED_ENVIRONMENT='+json.dumps({'record_ids': ids})+'\n')
        with contextlib.redirect_stdout(stream):
            for key in ids:
                arm, task = key.split(':')
                emit_record({'record_id': key, 'arm': arm, 'task_id': task,
                    'split': 'held-out-eval', 'success': False, 'failure_category': 'fixture'})
        stream.write('PAIRED_SUMMARY='+json.dumps({'run_id': 'fixture', 'record_ids': ids,
            'metrics': metrics, 'interpretation': 'Unit test; not model results'})+'\n')
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp)/'fixture.log'
            log.write_text(stream.getvalue())
            rows = extract(log, Path(temp)/'out', expected_count=20, expected_split='held-out-eval')
            self.assertEqual(len(rows), 20)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                extract(log, Path(temp)/'bad', expected_count=20)


class RecordedAuditTests(unittest.TestCase):
    def records(self):
        return [json.loads(line) for line in
            (ROOT/'research/results/tiny-aya-corrective-replay/20261114T222638Z.jsonl').read_text().splitlines()]

    def test_independent_contract_recomputation_matches_all_archived_episodes(self):
        for record in self.records():
            self.assertEqual(set(recorded_contract_failures(record)), set(record['criterion_failures']))

    def test_expected_failure_is_evidence_and_success_claim_cannot_override_contract(self):
        record = next(r for r in self.records() if r['task_id']=='B046')
        record['success_condition'] = {'validation_exit_codes': [1], 'no_task_file_changes': True}
        record['agent_stop_reason'] = 'completed'
        record['validation']['exit_code'] = 1
        self.assertEqual(recorded_contract_failures(record), [])
        record['success'] = True
        record['success_condition']['response_contains'] = ['missing-evidence-fixture']
        self.assertEqual(recorded_contract_failures(record), ['missing_response_term:missing-evidence-fixture'])

    @unittest.skipUnless(importlib.util.find_spec('auditkit'), 'optional AuditKIT core absent')
    def test_real_sdk_evaluates_archived_episodes_without_method_errors(self):
        from evals.auditkit_report import evaluate_records
        result = evaluate_records(self.records())
        self.assertEqual(len(result.rows), 8)
        self.assertTrue(all(row.outcome['verdict']=='failure' for row in result.rows))


if __name__=='__main__':
    unittest.main()
