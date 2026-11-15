import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_observation_intervention import extract
from training.agenttune.observation_intervention import emit
from training.agenttune.prepare_observation_intervention import REPLAY,matched_cases,prepare
from training.agenttune.prepare_sft import ROOT,read_rows


class ObservationInterventionTests(unittest.TestCase):
    def setUp(self):
        self.records=[r for r in read_rows(REPLAY) if r['arm']=='adapter' and r['task_id'] in ('B021','B026','B041')]
        self.source=(ROOT/'evals/benchmark/snapshots/miniapp/shop/config.py').read_text()

    def test_only_observation_stdout_changes_and_original_is_exact(self):
        for record in self.records:
            before=copy.deepcopy(record)
            original,full=matched_cases(record,self.source)
            i=2 if record['task_id']=='B026' else 1
            self.assertEqual(original['messages'],record['agenttune_trace']['model_prompts'][i])
            self.assertEqual(original['messages'][:-1],full['messages'][:-1])
            a=json.loads(original['messages'][-1]['content'].split('\n',1)[1])
            b=json.loads(full['messages'][-1]['content'].split('\n',1)[1])
            self.assertEqual(b,{**a,'stdout':self.source})
            self.assertEqual(full['identity_control'],record['task_id']=='B026')
            if record['task_id']=='B026':self.assertEqual(original['messages'],full['messages'])
            else:self.assertNotEqual(original['messages'],full['messages'])
            self.assertEqual(record,before)

    def test_wrong_source_and_non_support_record_are_rejected(self):
        with self.assertRaises(ValueError):matched_cases(self.records[0],'different file')
        with self.assertRaises(ValueError):matched_cases({**self.records[0],'split':'held-out-eval'},self.source)

    def test_bundle_has_only_six_inputs_and_no_targets_or_training_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle'
            manifest=prepare(root)
            self.assertEqual(manifest['case_count'],6)
            self.assertFalse(manifest['gold_targets_bundled'])
            self.assertEqual(len(read_rows(root/'data/cases.jsonl')),6)
            self.assertFalse((root/'evals').exists())
            self.assertFalse((root/'data/approved.jsonl').exists())
            self.assertFalse((root/'training/agenttune/sft_smoke.py').exists())
            with self.assertRaises(ValueError):prepare(root)

    def test_extractor_requires_all_chunks_and_roundtrips_large_prompts(self):
        identities=[f'{arm}:B{x}:{c}' for arm in ('base','adapter') for x in (21,26,41) for c in ('original','full_source')]
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            for key in identities:emit({'record_id':key,'source_split':'train-support','payload':'x'*20000})
        stream.write('OBSERVATION_INTERVENTION_ENVIRONMENT='+json.dumps({'record_ids':identities})+'\n')
        stream.write('OBSERVATION_INTERVENTION_SUMMARY='+json.dumps({'run_id':'unit-test','record_ids':identities})+'\n')
        with tempfile.TemporaryDirectory() as temp:
            log=Path(temp)/'job.log';log.write_text(stream.getvalue())
            extract(log,Path(temp)/'out')
            rows=read_rows(Path(temp)/'out/unit-test.jsonl')
            self.assertEqual(len(rows),12)
            self.assertTrue(all(len(r['payload'])==20000 for r in rows))
            log.write_text('\n'.join(stream.getvalue().splitlines()[1:]))
            with self.assertRaisesRegex(ValueError,'incomplete'):extract(log,Path(temp)/'bad')


if __name__=='__main__':unittest.main()
