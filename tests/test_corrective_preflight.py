import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_corrective_preflight import extract
from training.agenttune.corrective_preflight import emit,inspect_labels
from training.agenttune.prepare_corrective_preflight import prepare,verify_export
from training.agenttune.prepare_sft import ROOT,action_rows,read_rows
from training.agenttune.weighted_loss import action_spans,old_file_content


class CorrectivePreflightTests(unittest.TestCase):
    def test_labels_mask_archived_turns_and_supervise_both_closings(self):
        ids=[9,261001,6,3,7,261001,6];labels=[-100]*4+[7,261001,6]
        result=inspect_labels(ids,labels,ids[:4],[261001,6])
        self.assertTrue(result['all_prompt_labels_masked'])
        self.assertTrue(result['both_closing_tokens_supervised'])
        bad=inspect_labels(ids,[-100]*3+[3,7,261001,6],ids[:4],[261001,6])
        self.assertFalse(bad['all_prompt_labels_masked'])
        self.assertFalse(bad['prompt_equals_generation_prefix'])

    def test_complete_read_inference_rejects_partial_and_truncated_slices(self):
        def row(end,truncated=False):
            return {'prompt':[{'role':'assistant','content':json.dumps([{'tool_name':'read_file','parameters':{'path':'a.py','start_line':1,'end_line':end}}])},
                {'role':'user','content':'Tool result (JSON):\n'+json.dumps({'success':True,'stdout':'a\nb\n','metadata':{},'stdout_truncated':truncated})}]}
        self.assertEqual(old_file_content(row(3),'a.py'),'a\nb\n')
        with self.assertRaises(ValueError):old_file_content(row(2),'a.py')
        with self.assertRaises(ValueError):old_file_content(row(3,True),'a.py')

    def test_new_corrective_edits_have_weights_and_old_spans_are_unchanged(self):
        rows,_=verify_export()
        scheme=json.loads((ROOT/'configs/tiny_aya_weighted_sft.json').read_text())['loss_weighting']
        writes=[r for r in rows if json.loads(r['completion'][0]['content'])[0]['tool_name']=='write_file']
        self.assertEqual(len(writes),3)
        for row in writes:self.assertTrue(any(s['kind']=='changed_code' for s in action_spans(row,scheme)))
        gold=action_rows(read_rows(ROOT/'data/sft/gold_support.approved.jsonl'))
        old=json.loads((ROOT/'research/results/tiny-aya-weighted-sft/20261114T160511Z.result.json').read_text())
        for audit in old['training_audit']['weight_annotations']:
            spans=action_spans(gold[audit['action_index']],scheme)
            self.assertEqual(spans,[{k:v for k,v in s.items() if k!='label_positions'} for s in audit['spans']])

    def test_bundle_contains_admitted_targets_without_audit_only_or_heldout_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle';manifest=prepare(root)
            self.assertEqual(manifest['label_count'],15);self.assertEqual(manifest['generation_count'],12)
            self.assertFalse((root/'evals').exists())
            self.assertFalse((root/'data/source_records.jsonl').exists())
            self.assertFalse((root/'training/agenttune/run_sft_job.sh').exists())
            self.assertTrue(all('truthful-failure' not in r['source_example_id'] for r in read_rows(root/'data/rows.jsonl')))

    def test_transport_recovers_large_records_and_rejects_missing_chunks(self):
        identities=[f'label:{i}' for i in range(15)]+[f'generation:{i}' for i in range(12)]
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            for identity in identities:emit({'record_id':identity,'kind':identity.split(':')[0],'source_split':'train-support','payload':'x'*20000})
        for suffix,value in [('ENVIRONMENT',{'record_ids':identities}),('SUMMARY',{'run_id':'unit-test','record_ids':identities})]:
            stream.write('CORRECTIVE_PREFLIGHT_'+suffix+'='+json.dumps(value)+'\n')
        with tempfile.TemporaryDirectory() as temp:
            log=Path(temp)/'job.log';log.write_text(stream.getvalue())
            extract(log,Path(temp)/'out')
            self.assertEqual(len(read_rows(Path(temp)/'out/unit-test.label.jsonl')),15)
            log.write_text('\n'.join(stream.getvalue().splitlines()[1:]))
            with self.assertRaisesRegex(ValueError,'incomplete'):extract(log,Path(temp)/'bad')


if __name__=='__main__':unittest.main()
