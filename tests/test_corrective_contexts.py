import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_corrective_contexts import extract
from training.agenttune.corrective_contexts import aggregate,compare_action,emit
from training.agenttune.prepare_corrective_contexts import prepare
from training.agenttune.prepare_sft import read_rows


class CorrectiveContextTests(unittest.TestCase):
    def test_strict_action_comparison_and_complete_source_inspection(self):
        gold='[{"tool_name":"write_file","parameters":{"path":"a.py","content":"def f():\\n    return 1\\n"}}]'
        equivalent=json.dumps(json.loads(gold),indent=2)
        comparison=compare_action(equivalent,gold)
        self.assertTrue(comparison['exact_action_match'])
        self.assertEqual(comparison['defined_functions'],['f'])
        self.assertFalse(compare_action(gold+'\n'+gold,gold)['schema_valid'])
        wrong=json.loads(gold);wrong[0]['parameters']['content']='return 1'
        comparison=compare_action(json.dumps(wrong),gold)
        self.assertFalse(comparison['write_content_exact'])
        self.assertFalse(comparison['exact_action_match'])
        self.assertFalse(compare_action('[{"tool_name":"unknown","parameters":{}}]',gold)['schema_valid'])
        finish='[{"tool_name":"finish","parameters":{"answer":"Observed failure."}}]'
        paraphrase=finish.replace('Observed failure.','The test failed.')
        comparison=compare_action(paraphrase,finish)
        self.assertTrue(comparison['tool_name_match'])
        self.assertFalse(comparison['exact_action_match'])

    def test_bundle_contains_exact_admitted_rows_without_training_or_benchmark_launcher(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle';result=prepare(root)
            self.assertEqual(result['context_count'],15);self.assertEqual(result['prediction_count'],30)
            self.assertFalse((root/'evals').exists())
            self.assertFalse((root/'training/agenttune/sft_smoke.py').exists())
            self.assertFalse((root/'training/agenttune/run_sft_job.sh').exists())
            rows=read_rows(root/'data/rows.jsonl')
            self.assertTrue(all('truthful-failure' not in r['source_example_id'] for r in rows))
            self.assertTrue(all(r['source_message_index']>=r['context_message_count'] for r in rows))

    def test_transport_requires_all_contexts_and_predictions_and_rejects_missing_chunk(self):
        contexts=[f'context:{i:02d}' for i in range(15)]
        predictions=[f'{arm}:{i:02d}' for arm in ('base','adapter') for i in range(15)]
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            for kind,ids in [('context',contexts),('prediction',predictions)]:
                for identity in ids:
                    emit({'kind':kind,'record_id':identity,'source_split':'train-support','payload':'x'*15000})
        stream.write('CORRECTIVE_CONTEXT_ENVIRONMENT='+json.dumps({'run_id':'unit-test','context_ids':contexts,'prediction_ids':predictions})+'\n')
        stream.write('CORRECTIVE_CONTEXT_SUMMARY='+json.dumps({'run_id':'unit-test','context_count':15,'prediction_count':30})+'\n')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);log=root/'job.log';log.write_text(stream.getvalue())
            extract(log,root/'out')
            self.assertEqual(len(read_rows(root/'out/unit-test.context.jsonl')),15)
            self.assertEqual(len(read_rows(root/'out/unit-test.prediction.jsonl')),30)
            log.write_text('\n'.join(stream.getvalue().splitlines()[1:]))
            with self.assertRaises(ValueError):extract(log,root/'incomplete')

    def test_aggregate_keeps_teacher_forcing_separate_from_generated_action(self):
        values=[]
        for arm in ('base','adapter'):
            values.append({'arm':arm,'expected_tool':'shell','comparison':{'schema_valid':False,'tool_name_match':False,'exact_action_match':False},
                'teacher_forced':{'exact_sequence':True,'token_accuracy':1.0,'mean_nll':0.1},
                'generated_token_ids':[5,261001],'hit_token_cap':False})
        metrics=aggregate(values)
        self.assertEqual(metrics['adapter']['teacher_forced_exact'],1)
        self.assertEqual(metrics['adapter']['exact_action_matches'],0)


if __name__=='__main__':unittest.main()
