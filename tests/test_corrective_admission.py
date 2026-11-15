import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from training.agenttune.admit_corrective_support import ADMIT_IDS,EXCLUDE_IDS,admit,export_rows,sha_file,validate_review
from training.agenttune.prepare_sft import ROOT,read_rows


class CorrectiveAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        source=ROOT/'data/corrective/support_corrective.v1.jsonl'
        review=source.with_suffix('.review.pending.jsonl')
        if not review.exists():review=source.with_suffix('.review.jsonl')
        cls.source=cls.root/'batch.jsonl';shutil.copy2(source,cls.source)
        shutil.copy2(review,cls.source.with_suffix('.review.jsonl'))
        cls.receipt=cls.root/'receipt.json';shutil.copy2(source.with_suffix('.owner-review-receipt.json'),cls.receipt)
        cls.source_hash=sha_file(cls.source);cls.receipt_hash=sha_file(cls.receipt)
        cls.output=cls.root/'sft.jsonl'
        cls.manifest=admit(cls.source,cls.receipt,cls.output)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_only_four_reviewed_continuations_are_admitted(self):
        records=read_rows(self.source.with_suffix('.admitted.jsonl'))
        self.assertEqual({r['example_id'] for r in records},ADMIT_IDS)
        rows=read_rows(self.output)
        self.assertEqual(len(rows),15)
        self.assertTrue(all(r['source_example_id'] not in EXCLUDE_IDS for r in rows))
        self.assertTrue(all(r['training_eligible'] and not r['review_required'] for r in rows))
        self.assertTrue(all(r['source_message_index']>=r['context_message_count'] for r in rows))
        self.assertEqual(self.manifest['duplicate_count'],0)
        self.assertEqual(self.manifest['revalidated_count'],4)
        self.assertFalse(self.manifest['training_performed'])

    def test_reviews_applied_without_rewriting_receipt_or_candidates(self):
        self.assertEqual(sha_file(self.source),self.source_hash)
        self.assertEqual(sha_file(self.receipt),self.receipt_hash)
        self.assertTrue(all(r['decision']=='approved' for r in read_rows(self.source.with_suffix('.review.jsonl'))))
        backup=self.source.with_suffix('.review.pending.jsonl')
        receipt=json.loads(self.receipt.read_text())
        self.assertEqual(sha_file(backup),receipt['review_sidecar_sha256_at_receipt'])
        self.assertTrue(all(r['decision']=='pending' for r in read_rows(backup)))

    def test_changed_data_missing_approval_and_wrong_scope_fail_closed(self):
        records=read_rows(self.source)
        receipt=json.loads(self.receipt.read_text())
        reviews=read_rows(self.source.with_suffix('.review.pending.jsonl'))
        changed=copy.deepcopy(records);changed[0]['messages'][1]['content']='changed task'
        with self.assertRaises(ValueError):validate_review(changed,receipt,reviews)
        rejected=copy.deepcopy(receipt);rejected['decisions'][0]['decision']='rejected'
        with self.assertRaises(ValueError):validate_review(records,rejected,reviews)
        wrong=copy.deepcopy(receipt);wrong['decisions'][-1]['scope']='review_only_admission_pending'
        with self.assertRaisesRegex(ValueError,'permanently excluded'):validate_review(records,wrong,reviews)
        with self.assertRaisesRegex(ValueError,'excluded'):
            export_rows([{'example_id':next(iter(EXCLUDE_IDS)),'admission':{'status':'admitted'}}])

    def test_prefix_boundaries_and_expected_failure_remain_correct(self):
        checks=read_rows(self.source.with_suffix('.revalidation.jsonl'))
        self.assertTrue(all(r['verified'] for r in checks))
        explanation=next(r for r in checks if r['example_id']=='C-B046-explain-evidence-v1')
        self.assertEqual(explanation['validation']['exit_code'],1)
        self.assertEqual(explanation['changed_files'],[])
        self.assertIn('10 != 25',explanation['validation']['stderr'])
        boundaries=read_rows(self.output.with_suffix('.boundaries.jsonl'))
        self.assertEqual(len(boundaries),15)
        self.assertTrue(all(r['supervise_only_completion'] and not r['supervise_prompt'] for r in boundaries))
        self.assertTrue(all(r['tokenizer_label_audit']=='pending' for r in boundaries))
        with self.assertRaises(FileExistsError):admit(self.source,self.receipt,self.output)


if __name__=='__main__':unittest.main()
