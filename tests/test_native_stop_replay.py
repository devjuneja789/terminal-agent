import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training.agenttune.paired_replay import capture_generations,generation_stops
from training.agenttune.prepare_native_stop_replay import TRAINING,prepare
from training.agenttune.prepare_replay_bundle import prepare as legacy_prepare


class NativeStopReplayTests(unittest.TestCase):
    def test_legacy_default_is_unchanged_and_explicit_union_preserves_all_native_ids(self):
        self.assertEqual(generation_stops({},SimpleNamespace(eos_token_id=6),3),6)
        tokenizer=SimpleNamespace(eos_token_id=6,unk_token_id=0,convert_tokens_to_ids=lambda _:261001)
        self.assertEqual(generation_stops({'eos_token_ids':[3,6,261001]},tokenizer,3),[3,6,261001])
        for stops in ([6,261001],[3,6],[3,6,261001,3],[3,True,261001],[[3],6,261001]):
            with self.assertRaises(ValueError):generation_stops({'eos_token_ids':stops},tokenizer,3)

    def test_new_version_changes_only_opt_in_stopping_and_telemetry(self):
        with tempfile.TemporaryDirectory() as temp:
            old,new=Path(temp)/'old',Path(temp)/'new'
            legacy_prepare(old,TRAINING);result=prepare(new)
            before=json.loads((old/'configs/tiny_aya_support_rollout.json').read_text())
            after=json.loads((new/'configs/tiny_aya_support_rollout.json').read_text())
            self.assertEqual(after['config_id'],'terminal-slm-native-stops-support-replay-v1')
            after['config_id']=before['config_id']
            for key in ('eos_token_ids','capture_stop_audit','stop_policy_version','source_stop_preflight_run_id'):
                after['replay'].pop(key)
            self.assertEqual(after,before)
            self.assertEqual((old/'evals/benchmark/train_support.jsonl').read_bytes(),(new/'evals/benchmark/train_support.jsonl').read_bytes())
            self.assertFalse((new/'evals/benchmark/heldout_eval.jsonl').exists())
            self.assertFalse((new/'data').exists())
            self.assertEqual(result['eos_token_ids'],[3,6,261001])

    def test_generation_audit_forwards_args_and_returns_exact_output(self):
        class Values:
            def __init__(self,ids):self.ids=ids
            def tolist(self):return list(self.ids)
        class Matrix:
            shape=(1,4)
            def __getitem__(self,key):return Values([5,6,10,261001][key[1]])
        class Model:
            generation_config=SimpleNamespace(eos_token_id=[3,6,261001])
            def generate(self,*args,**kwargs):
                self.seen=(args,kwargs);return output
        output=Matrix();model=Model();prompt=SimpleNamespace(shape=(1,2))
        audit=capture_generations(model,256)
        result=model.generate(input_ids=prompt,max_new_tokens=256,use_cache=True,do_sample=False)
        self.assertIs(result,output)
        self.assertEqual(model.seen,((),{'input_ids':prompt,'max_new_tokens':256,'use_cache':True,'do_sample':False}))
        self.assertEqual(audit[0]['generated_token_ids'],[10,261001])
        self.assertTrue(audit[0]['stopped_on_configured_id'])
        self.assertFalse(audit[0]['hit_token_cap'])
        self.assertEqual(audit[0]['configured_stop_ids'],[3,6,261001])


if __name__=='__main__':unittest.main()
