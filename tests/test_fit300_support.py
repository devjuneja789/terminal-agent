import json
import tempfile
import unittest
from pathlib import Path

from training.agenttune.prepare_fit300_support import prepare
from training.agenttune.prepare_sft import ROOT


class Fit300SupportTests(unittest.TestCase):
    def test_same_runtime_new_adapter_and_support_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'bundle';prepare(root)
            actual=json.loads((root/'configs/tiny_aya_support_rollout.json').read_text())
            previous=json.loads((ROOT/'research/results/tiny-aya-native-stop-replay/20261114T220329Z.metadata.json').read_text())['config']
            self.assertEqual(actual['replay']['task_ids'],['B021','B026','B041','B046'])
            self.assertEqual(actual['replay']['adapter_revision'],'1eed40612640c78d88d9fe0cac36e06b1148d1fd')
            for key in ('model','generation','agent','prompt','hashes','rollout_controls'):
                self.assertEqual(actual[key],previous[key])
            self.assertFalse((root/'evals/benchmark/heldout_eval.jsonl').exists())
            self.assertFalse((root/'data').exists())


if __name__=='__main__':unittest.main()
