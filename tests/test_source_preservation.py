import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_source_preservation import analyze, validate_content
from training.agenttune.prepare_action_diagnostic import prepare
from training.agenttune.prepare_sft import ROOT, action_rows, read_rows
from training.agenttune.source_preservation import inspect_write_prediction


class SourcePreservationTests(unittest.TestCase):
    def setUp(self):
        rows=action_rows(read_rows(ROOT/'data/sft/gold_support.approved.jsonl'))
        self.row=next(r for r in rows if r['source_task_id']=='B041' and json.loads(r['completion'][0]['content'])[0]['tool_name']=='write_file')
        self.gold=self.row['completion'][0]['content']

    def action(self,content):
        action=json.loads(self.gold)
        action[0]['parameters']['content']=content
        return json.dumps(action)

    def test_approved_write_preserves_complete_module(self):
        before=copy.deepcopy(self.row)
        score=inspect_write_prediction(self.gold,self.row)
        self.assertTrue(score['gold_content_exact'])
        self.assertTrue(score['gold_ast_equal'])
        self.assertTrue(score['full_file_preserved'])
        self.assertEqual(score['damaged_unchanged_nodes'],[])
        self.assertEqual(self.row,before)

    def test_formatting_changes_are_not_missing_code(self):
        content=json.loads(self.gold)[0]['parameters']['content']
        score=inspect_write_prediction(self.action(ast.unparse(ast.parse(content))),self.row)
        self.assertTrue(score['gold_ast_equal'])
        self.assertTrue(score['full_file_preserved'])
        self.assertFalse(score['gold_content_exact'])

    def test_snippet_and_unrelated_change_fail_preservation(self):
        snippet=inspect_write_prediction(self.action('DEFAULT_TIMEOUT_SECONDS = 15\nrequest_timeout()\n'),self.row)
        self.assertTrue(snippet['python_parse_valid'])
        self.assertFalse(snippet['full_file_preserved'])
        self.assertIn('FunctionDef:request_timeout',snippet['missing_original_nodes'])
        content=json.loads(self.gold)[0]['parameters']['content'].replace('MAX_PAGE_SIZE = 100','MAX_PAGE_SIZE = 1')
        score=inspect_write_prediction(self.action(content),self.row)
        self.assertFalse(score['full_file_preserved'])
        self.assertTrue(score['damaged_unchanged_nodes'])

    def test_invalid_syntax_wrong_tool_and_wrong_path_are_separate(self):
        score=inspect_write_prediction(self.action('def broken('),self.row)
        self.assertTrue(score['eligible_write'])
        self.assertFalse(score['python_parse_valid'])
        for text in ['not json','[{"tool_name":"shell","parameters":{"command":"true"}}]',self.gold.replace('shop/config.py','other.py')]:
            self.assertFalse(inspect_write_prediction(text,self.row)['eligible_write'])

    def test_weighted_bundle_preserves_actual_source_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle'
            prepare(root,ROOT/'research/results/tiny-aya-weighted-sft/20261114T160511Z.result.json')
            config=json.loads((root/'configs/tiny_aya_sft_smoke.json').read_text())
            self.assertEqual(config['experiment_kind'],'support_weighted_objective')
            self.assertFalse((root/'evals').exists())
            self.assertFalse((root/'training/agenttune/run_sft_job.sh').exists())

    def test_analysis_rejects_different_adapter_lineage(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'lineage'):
                analyze(ROOT/'research/results/tiny-aya-action-diagnostic/20261114T152447Z',Path(temp)/'bad')

    def test_validator_uses_disposable_snapshot_and_distinguishes_broken_snippet(self):
        from evals.benchmark.loader import load_benchmark
        task=next(t for t in load_benchmark('train-support') if t.task_id=='B041')
        before=(task.snapshot_path/'shop/config.py').read_bytes()
        good=json.loads(self.gold)[0]['parameters']['content']
        result=validate_content(task,good,'shop/config.py')
        self.assertTrue(result['passed'])
        self.assertEqual(result['changed_files'],['shop/config.py'])
        broken=validate_content(task,'DEFAULT_TIMEOUT_SECONDS = 15\nrequest_timeout()\n','shop/config.py')
        self.assertFalse(broken['passed'])
        self.assertIn('NameError',broken['stderr'])
        self.assertEqual((task.snapshot_path/'shop/config.py').read_bytes(),before)

    def test_named_lower_bound_pass_does_not_hide_upper_bound_regression(self):
        from evals.benchmark.loader import load_benchmark
        task=next(t for t in load_benchmark('train-support') if t.task_id=='B021')
        content=json.loads(self.gold)[0]['parameters']['content'].replace('DEFAULT_TIMEOUT_SECONDS = 15','DEFAULT_TIMEOUT_SECONDS = 10').replace('min(value, MAX_PAGE_SIZE)','max(value, 1)')
        self.assertTrue(validate_content(task,content,'shop/config.py')['passed'])
        extra=validate_content(task,content,'shop/config.py',validation_command=('python3','-m','unittest','tests.test_contracts.Contracts.test_page_size_upper_bound'))
        self.assertFalse(extra['passed'])
        self.assertIn('101 != 100',extra['stderr'])


if __name__=='__main__':unittest.main()
