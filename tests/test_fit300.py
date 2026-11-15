import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.freeze_submission_candidate import score
from training.agenttune.prepare_fit300 import prepare,PROFILE
from training.agenttune.prepare_joint_training import validate_joint_result
from training.agenttune.prepare_sft import ROOT,read_rows


class FitBudgetTests(unittest.TestCase):
    def test_only_horizon_changes_and_sources_stay_fixed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle';prepare(root,'test/fit300')
            new=json.loads((root/'configs/tiny_aya_sft_smoke.json').read_text())
            old=json.loads((ROOT/'configs/tiny_aya_joint_choice_sft.json').read_text())
            self.assertEqual({k for k in new if new[k]!=old.get(k)},
                {'config_id','max_steps','expected_epochs','artifact_repo'})
            self.assertEqual(new['max_steps'],300);self.assertEqual(new['expected_epochs'],20)
            self.assertEqual(read_rows(root/'data/corrective_rows.jsonl'),read_rows(ROOT/'data/sft/tiny_aya_corrective_v1.jsonl'))
            self.assertFalse((root/'evals').exists())

    def test_validator_requires_fixed_horizon(self):
        result=json.loads((ROOT/'research/results/tiny-aya-joint-sft/20261115T062137Z.result.json').read_text())
        result['config']=json.loads(PROFILE.read_text());result['config']['artifact_repo']='test/fit300'
        with self.assertRaises(ValueError):validate_joint_result(result,config_path=PROFILE)
        # Temporary unit fixture, not a training result or evidence.
        result['training_audit']['global_step']=300;result['training_audit']['epoch']=20
        validate_joint_result(result,config_path=PROFILE)
        broken=copy.deepcopy(result);broken['training_audit']['skipped_optimizer_steps']=[100]
        with self.assertRaises(ValueError):validate_joint_result(broken,config_path=PROFILE)

    def test_selection_priority_and_tie_are_explicit(self):
        def metric(exact,finish,schema,tool):
            return {'exact_action_matches':exact,'schema_valid':schema,'tool_name_matches':tool,
                'by_expected_tool':{'finish':{'tool_name_matches':finish}}}
        baseline=score(metric(3,0,14,8))
        self.assertGreater(score(metric(4,0,10,5)),baseline)
        self.assertGreater(score(metric(3,1,14,8)),baseline)
        self.assertEqual(score(metric(3,0,14,8)),baseline)


if __name__=='__main__':unittest.main()
