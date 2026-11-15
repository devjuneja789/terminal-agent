import copy
import json
import tempfile
import unittest
from pathlib import Path

from training.agenttune.tool_choice_objective import choice_weights
from training.agenttune.prepare_tool_choice_training import prepare,validate_choice_result,PROFILE
from training.agenttune.prepare_tool_choice_contexts import prepare as prepare_contexts
from training.agenttune.prepare_sft import ROOT,read_rows


class ToolChoiceObjectiveTests(unittest.TestCase):
    def test_only_native_tool_name_positions_have_positive_weight(self):
        labels=[-100,-100,10,11,12,261001,6]
        audit={'spans':[{'kind':'tool_name','label_positions':[2,3]},
            {'kind':'json_closure','label_positions':[4]}],
            'label_positions':[2,3,4,5,6],'target_weights':[8.,8.,4.,1.,1.]}
        original=copy.deepcopy(audit)
        weights,result=choice_weights(labels,audit)
        self.assertEqual(weights,[0.,0.,1.,1.,0.,0.,0.])
        self.assertEqual(result['target_weights'],[1.,1.,0.,0.,0.])
        self.assertEqual(result['normalizer'],2)
        self.assertEqual(audit,original)
        for positions in ([0],[5],[],[99]):
            bad=copy.deepcopy(audit);bad['spans'][0]['label_positions']=positions
            with self.assertRaises(ValueError):choice_weights(labels,bad)

    def test_bundle_retains_exact_data_and_changes_only_objective_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle';manifest=prepare(root,'test/tool-choice')
            self.assertEqual(manifest['action_target_count'],15)
            self.assertEqual(read_rows(root/'data/corrective_rows.jsonl'),read_rows(ROOT/'data/sft/tiny_aya_corrective_v1.jsonl'))
            self.assertFalse((root/'evals').exists())
            config=json.loads((root/'configs/tiny_aya_sft_smoke.json').read_text())
            old=json.loads((ROOT/'configs/tiny_aya_corrective_sft.json').read_text())
            self.assertEqual({k for k in config if config[k]!=old.get(k)},
                {'config_id','experiment_kind','objective','artifact_repo','amp_initial_scale','gradient_preflight_all_batches'})
            self.assertEqual(config['max_steps'],60)
            self.assertTrue((root/'training/agenttune/tool_choice_objective.py').exists())

    def test_old_objective_is_not_admitted_as_tool_choice_training(self):
        old=json.loads((ROOT/'research/results/tiny-aya-corrective-sft/20261114T222003Z.result.json').read_text())
        with self.assertRaises(ValueError):validate_choice_result(old)

    def test_context_bundle_preserves_procedure_and_requires_choice_only_weights(self):
        # Synthetic status fields belong only to this temporary unit fixture.
        result=json.loads((ROOT/'research/results/tiny-aya-corrective-sft/20261114T222003Z.result.json').read_text())
        result['config']=json.loads(PROFILE.read_text());result['config']['artifact_repo']='test/choice-fixture'
        result['loss']='tool_choice_only'
        result['training_audit']['tool_choice_objective_self_check']={
            'nonchoice_logit_invariance':True,'only_selected_causal_position_has_gradient':True,'model_parameters_updated':False,
            'gradient_preflight':{'scale':256,'batch_count':15,'optimizer_updates':0,'all_gradients_finite':True,'gradients_cleared':True}}
        annotations=[]
        for a in result['training_audit']['weight_annotations']:
            labels=[-100]*(max(a['label_positions'])+1)
            for p,token in zip(a['label_positions'],a['target_token_ids'],strict=True):labels[p]=token
            _,a=choice_weights(labels,a);annotations.append(a)
        result['training_audit']['weight_annotations']=annotations
        validate_choice_result(result)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'unit-fixture.json';source.write_text(json.dumps(result))
            prepare_contexts(root/'bundle',source)
            config=json.loads((root/'bundle/configs/corrective_contexts.json').read_text())
            control=json.loads((ROOT/'research/results/tiny-aya-corrective-contexts/20261114T224529Z.metadata.json').read_text())['config']
            config.pop('objective')
            for k in ('config_id','source_training_run_id','adapter_repo','adapter_revision','adapter_subfolder','source_sha256'):
                config[k]=control[k]
            self.assertEqual(config,control)
            self.assertFalse((root/'bundle/evals').exists())
            self.assertFalse((root/'bundle/training/agenttune/sft_smoke.py').exists())
        broken=copy.deepcopy(result)
        broken['training_audit']['weight_annotations'][0]['target_weights'][-1]=1.
        with self.assertRaises(ValueError):validate_choice_result(broken)


if __name__=='__main__':unittest.main()
