import base64
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_action_diagnostic import extract
from training.agenttune.action_diagnostic import emit, inspect_boundary, match_prepared_row, parse_generated
from training.agenttune.prepare_action_diagnostic import prepare
from training.agenttune.prepare_sft import ROOT


class ActionDiagnosticTests(unittest.TestCase):
    def test_boundary_detects_misaligned_prefix_and_unmasked_context(self):
        audit=inspect_boundary([1,2,3,4],[-100,-100,3,4],[1,2])
        self.assertTrue(audit['training_prefix_equals_generation_prompt'])
        self.assertTrue(audit['contiguous_completion_suffix'])
        audit=inspect_boundary([1,2,3,4],[-100,2,-100,4],[1,2])
        self.assertFalse(audit['training_prefix_equals_generation_prompt'])
        self.assertFalse(audit['contiguous_completion_suffix'])
        with self.assertRaises(ValueError):inspect_boundary([1],[-100],[])

    def test_shuffled_prepared_rows_are_matched_by_messages(self):
        rows=[{'prompt':[{'content':'first'}],'completion':[{'content':'a'}]},
              {'prompt':[{'content':'second'}],'completion':[{'content':'b'}]}]
        self.assertEqual(match_prepared_row(rows[1],rows),1)
        with self.assertRaises(ValueError):match_prepared_row(rows[0],[rows[0],rows[0]])

    def test_action_comparison_distinguishes_schema_tool_and_full_match(self):
        gold='[{"tool_name":"read_file","parameters":{"path":"shop/config.py"}}]'
        self.assertTrue(parse_generated(gold,gold)['exact_action_match'])
        changed='[{"tool_name":"read_file","parameters":{"path":"other.py"}}]'
        self.assertTrue(parse_generated(changed,gold)['tool_name_match'])
        self.assertFalse(parse_generated(changed,gold)['exact_action_match'])
        self.assertFalse(parse_generated(gold+' prose',gold)['schema_valid'])
        self.assertFalse(parse_generated('[{"tool_name":"unknown","parameters":{}}]',gold)['schema_valid'])

    def test_bundle_has_only_approved_references_and_no_training_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle'
            manifest=prepare(root,ROOT/'research/results/tiny-aya-memorization/20261114T145152Z.result.json')
            self.assertEqual(manifest['action_count'],16)
            self.assertFalse((root/'evals').exists())
            self.assertFalse((root/'training/agenttune/run_sft_job.sh').exists())
            self.assertTrue((root/'training/agenttune/run_action_diagnostic_job.sh').is_file())

    def _log(self):
        stream=io.StringIO()
        bids=[f'b{i}' for i in range(16)]
        pids=[f'p{i}' for i in range(32)]
        with contextlib.redirect_stdout(stream):
            for kind,ids in [('boundary',bids),('prediction',pids)]:
                for identity in ids:
                    emit(kind,identity,{'record_id':identity,'source_split':'train-support','payload':'x'*15000})
        stream.write('ACTION_DIAGNOSTIC_ENVIRONMENT='+json.dumps({'boundary_ids':bids,'record_ids':pids})+'\n')
        stream.write('ACTION_DIAGNOSTIC_SUMMARY='+json.dumps({'run_id':'unit-test'})+'\n')
        return stream.getvalue()

    def test_extraction_is_lossless_and_requires_all_records(self):
        with tempfile.TemporaryDirectory() as temp:
            log=Path(temp)/'run.log';log.write_text(self._log())
            extract(log,Path(temp)/'out')
            rows=[json.loads(line) for line in (Path(temp)/'out/unit-test.prediction.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),32)
            self.assertTrue(all(len(r['payload'])==15000 for r in rows))
            log.write_text('\n'.join(self._log().splitlines()[1:]))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                extract(log,Path(temp)/'bad')


if __name__=='__main__':unittest.main()
