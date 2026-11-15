import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from training.agenttune.joint_choice_objective import joint_weights
from training.agenttune.prepare_joint_training import prepare,validate_joint_result,PROFILE
from training.agenttune.prepare_sft import ROOT,read_rows


class JointChoiceTests(unittest.TestCase):
    def test_mixture_keeps_completion_and_closing_tokens_positive(self):
        labels=[-100,-100,10,11,261001,6]
        audit={'spans':[{'kind':'tool_name','label_positions':[2,3]}],
            'label_positions':[2,3,4,5],'target_weights':[8.,8.,1.,1.]}
        before=copy.deepcopy(audit)
        weights,a=joint_weights(labels,[0.,0.,8.,8.,1.,1.],audit)
        self.assertTrue(math.isclose(sum(weights),1.))
        self.assertEqual(weights[:2],[0.,0.]);self.assertTrue(all(w>0 for w in weights[2:]))
        self.assertAlmostEqual(weights[2],.75+.25*8/18)
        self.assertAlmostEqual(weights[3],.25*8/18)
        self.assertEqual(a['decision_position'],2);self.assertEqual(audit,before)
        for share in (0,1,-1,float('nan')):
            with self.assertRaises(ValueError):joint_weights(labels,[0.,0.,8.,8.,1.,1.],audit,share)

    def test_bundle_has_same_15_targets_and_same_numerical_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle';manifest=prepare(root,'test/joint-fixture')
            self.assertEqual(manifest['action_target_count'],15)
            self.assertEqual(read_rows(root/'data/corrective_rows.jsonl'),read_rows(ROOT/'data/sft/tiny_aya_corrective_v1.jsonl'))
            self.assertFalse((root/'evals').exists())
            cfg=json.loads((root/'configs/tiny_aya_sft_smoke.json').read_text())
            old=json.loads((ROOT/'configs/tiny_aya_tool_choice_sft.json').read_text())
            self.assertEqual({k for k in cfg if cfg[k]!=old.get(k)},
                {'config_id','experiment_kind','objective','decision_share','artifact_repo'})

    def test_result_requires_actual_joint_coefficients_and_gradient_gates(self):
        r=json.loads((ROOT/'research/results/tiny-aya-tool-choice-sft/20261115T055053Z.result.json').read_text())
        with self.assertRaises(ValueError):validate_joint_result(r)
        # Explicit unit-test fixture only, never a research/model result.
        r['config']=json.loads(PROFILE.read_text());r['config']['artifact_repo']='test/joint-fixture'
        r['loss']='joint_first_choice_completion'
        r['training_audit']['joint_objective_self_check']={'prompt_logits_masked':True,'completion_logits_supervised':True,
            'decision_logits_emphasized':True,'model_parameters_updated':False,
            'gradient_preflight':r['training_audit']['tool_choice_objective_self_check']['gradient_preflight']}
        control=json.loads((ROOT/'research/results/tiny-aya-corrective-sft/20261114T222003Z.result.json').read_text())
        annotations=[]
        for a in control['training_audit']['weight_annotations']:
            labels=[-100]*(max(a['label_positions'])+1);weights=[0.]*len(labels)
            for p,t,w in zip(a['label_positions'],a['target_token_ids'],a['target_weights'],strict=True):labels[p]=t;weights[p]=w
            _,a=joint_weights(labels,weights,a);annotations.append(a)
        r['training_audit']['weight_annotations']=annotations;validate_joint_result(r)
        r['training_audit']['weight_annotations'][0]['target_weights'][-1]=0.
        with self.assertRaises(ValueError):validate_joint_result(r)


if __name__=='__main__':unittest.main()
