import copy
import json
import tempfile
import unittest
from pathlib import Path
from training.agenttune.prepare_task_start_training import prepare
from training.agenttune.task_start_data import verified_rows,ROWS_SHA256
from training.agenttune.prepare_sft import read_rows,digest
from training.agenttune.sft_smoke import verify_edit_target_coverage


class TaskStartCurriculumTests(unittest.TestCase):
    def test_mixed_six_edits_and_legacy_three_edits_are_checked(self):
        edits=[{'expected_tool':'write_file','spans':[{'kind':'changed_code'}]} for _ in range(6)]
        verify_edit_target_coverage(edits,mixed=True)
        verify_edit_target_coverage(edits[:3])
        with self.assertRaises(RuntimeError):verify_edit_target_coverage(edits[:3],mixed=True)
        with self.assertRaises(RuntimeError):verify_edit_target_coverage(edits)
        edits[-1]['spans']=[]
        with self.assertRaises(RuntimeError):verify_edit_target_coverage(edits,mixed=True)

    def test_sources_initial_targets_and_mask_boundaries_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle';manifest=prepare(root,'fixture/adapter')
            rows=read_rows(root/'data/corrective_rows.jsonl');refs=read_rows(root/'data/approved.jsonl')
            proof=root/'data/task_start_proofs'
            self.assertEqual(verified_rows(refs,rows,proof),rows)
            self.assertEqual(digest(rows),ROWS_SHA256)
            self.assertEqual(sum(len(r['prompt'])==2 for r in rows),4)
            self.assertEqual(len(rows),31)
            self.assertNotIn('label_gate',manifest)
            self.assertFalse((root/'evals').exists())
            config=json.loads((root/'configs/tiny_aya_sft_smoke.json').read_text())
            self.assertEqual(config['max_steps'],620)
            for r in rows:
                if r['source_family']=='admitted_corrective':
                    self.assertGreaterEqual(r['source_message_index'],r['context_message_count'])
            bad=copy.deepcopy(rows);bad[0]['completion'][0]['content']='[]'
            with self.assertRaises(ValueError):verified_rows(refs,bad,proof)
            bad=copy.deepcopy(refs);bad[0]['human_review']['decision']='rejected'
            with self.assertRaises(ValueError):verified_rows(bad,rows,proof)


if __name__=='__main__':unittest.main()
