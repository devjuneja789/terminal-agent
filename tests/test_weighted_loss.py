import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from training.agenttune.prepare_sft import ROOT, action_rows, read_rows
from training.agenttune.prepare_sft_bundle import prepare
from training.agenttune.weighted_loss import action_spans, align_weights, causal_weighted_loss, old_file_content, weighted_collator, initialize_loss_scale


class WeightAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.rows=action_rows(read_rows(ROOT/'data/sft/gold_support.approved.jsonl'))
        self.scheme=json.loads((ROOT/'configs/tiny_aya_weighted_sft.json').read_text())['loss_weighting']

    def test_approved_text_is_unchanged_and_all_three_edits_have_spans(self):
        original=copy.deepcopy(self.rows)
        writes=0
        for row in self.rows:
            spans=action_spans(row,self.scheme)
            self.assertTrue(any(s['kind']=='tool_name' for s in spans))
            self.assertTrue(any(s['kind']=='json_closure' for s in spans))
            if json.loads(row['completion'][0]['content'])[0]['tool_name']=='write_file':
                writes+=1
                self.assertTrue(any(s['kind']=='changed_code' for s in spans))
        self.assertEqual(writes,3)
        self.assertEqual(self.rows,original)

    def test_encoded_changed_spans_cover_15_and_keep_context_masked(self):
        row=next(r for r in self.rows if r['source_task_id']=='B041' and '"tool_name":"write_file"' in r['completion'][0]['content'])
        gold=row['completion'][0]['content'];spans=action_spans(row,self.scheme)
        changed=[gold[s['start']:s['end']] for s in spans if s['kind']=='changed_code']
        self.assertIn('5',''.join(changed))
        # Character-token fixture exercises masking, overlap=max and EOS weight.
        labels=[-100]*3+[1]*len(gold)+[6]
        offsets=[(0,0)]*3+[(i,i+1) for i in range(len(gold))]+[(0,0)]
        weights,audit=align_weights(labels,offsets,spans,0)
        self.assertEqual(weights[:3],[0,0,0]);self.assertEqual(weights[-1],1)
        self.assertEqual(max(weights),8)
        self.assertEqual(audit['normalizer'],sum(weights))

    def test_changed_code_without_full_prior_observation_is_rejected(self):
        row=next(r for r in self.rows if '"tool_name":"write_file"' in r['completion'][0]['content'])
        with self.assertRaisesRegex(ValueError,'complete file'):
            old_file_content({**row,'prompt':row['prompt'][:2]},'shop/config.py')

    def test_weighted_bundle_keeps_same_approved_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle'
            manifest=prepare(root,'owner/weighted',ROOT/'configs/tiny_aya_weighted_sft.json')
            self.assertEqual(manifest['action_target_count'],16)
            self.assertTrue((root/'training/agenttune/weighted_loss.py').is_file())
            self.assertFalse((root/'evals').exists())

    def test_escaped_changed_code_maps_back_to_json_string(self):
        old, new = 'x = 1\n', 'x = "a"\n'
        row={'prompt':[
            {'role':'assistant','content':json.dumps([{'tool_name':'read_file','parameters':{'path':'a.py','start_line':1,'end_line':10}}])},
            {'role':'user','content':'Tool result:\n'+json.dumps({'success':True,'stdout':old,'metadata':{'line_count':1}})}],
            'completion':[{'role':'assistant','content':json.dumps([{'tool_name':'write_file','parameters':{'path':'a.py','content':new}}])}]}
        gold=row['completion'][0]['content']
        spans=action_spans(row,self.scheme)
        changed=''.join(gold[s['start']:s['end']] for s in spans if s['kind']=='changed_code')
        self.assertEqual(changed, '\\"a\\"')

    def test_amp_scale_changes_only_numerical_startup_scale(self):
        from types import SimpleNamespace
        class Scaler:
            state={'scale':65536.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000,'_growth_tracker':0}
            def is_enabled(self):return True
            def state_dict(self):return dict(self.state)
            def load_state_dict(self,state):self.state=dict(state)
        audit=initialize_loss_scale(SimpleNamespace(scaler=Scaler()),8192)
        self.assertEqual(audit['before']['scale'],65536)
        self.assertEqual(audit['after'],{**audit['before'],'scale':8192.})


@unittest.skipUnless(importlib.util.find_spec('torch'),'numeric tensor tests require optional torch')
class WeightedMathTests(unittest.TestCase):
    def test_unit_weights_match_cross_entropy_value_and_gradients(self):
        import torch
        import torch.nn.functional as F
        values=torch.tensor([[[0.,1.,2.],[2.,1.,0.],[1.,0.,2.],[0.,0.,0.]]],requires_grad=True)
        labels=torch.tensor([[-100,-100,0,2]])
        weights=(labels!=-100).float()
        loss,plain,_=causal_weighted_loss(values,labels,weights)
        expected=F.cross_entropy(values[:,:-1,:].reshape(-1,3),labels[:,1:].reshape(-1),ignore_index=-100)
        self.assertTrue(torch.allclose(loss,expected));self.assertTrue(torch.allclose(plain,expected))
        a=torch.autograd.grad(loss,values,retain_graph=True)[0]
        b=torch.autograd.grad(expected,values)[0]
        self.assertTrue(torch.allclose(a,b));self.assertEqual(float(a[0,0].abs().sum()),0)

    def test_weights_change_relative_gradient_without_unmasking_context(self):
        import torch
        import torch.nn.functional as F
        logits=torch.zeros(1,4,3,requires_grad=True)
        labels=torch.tensor([[-100,-100,1,2]])
        weights=torch.tensor([[0.,0.,8.,1.]])
        loss,plain,normalizer=causal_weighted_loss(logits,labels,weights)
        self.assertEqual(normalizer,9)
        loss.backward()
        ratio=float(logits.grad[0,1].abs().sum()/logits.grad[0,2].abs().sum())
        self.assertAlmostEqual(ratio,8,places=5)
        self.assertEqual(float(logits.grad[0,0].abs().sum()),0)
        with self.assertRaises(ValueError):causal_weighted_loss(logits,labels,torch.ones_like(weights))

    def test_collator_preserves_context_mask_and_right_padding(self):
        import torch
        def original(features):
            self.assertNotIn('loss_weights',features[0])
            return {'labels':torch.tensor([[-100,2,6],[-100,6,-100]])}
        batch=weighted_collator(original)([{'loss_weights':[0,8,1]},{'loss_weights':[0,1]}])
        self.assertEqual(batch['loss_weights'].tolist(),[[0,8,1],[0,1,0]])


if __name__=='__main__':unittest.main()
