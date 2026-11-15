"""Stage a separately versioned support-only replay with the native stop union."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from training.agenttune.prepare_replay_bundle import prepare as prepare_legacy
from training.agenttune.prepare_sft import ROOT,digest

TRAINING=ROOT/'research/results/tiny-aya-weighted-sft/20261114T160511Z.result.json'


def prepare(destination: Path,training_result: Path | None=None) -> dict:
    source=training_result or TRAINING
    training=json.loads(source.read_text())
    if training['artifact_commit']!='1c0559e7eeab4aaea8fffeddf91ff741cfdb2fe0':
        raise ValueError('this replay is fixed to the EXP-012 weighted adapter')
    candidate=json.loads((ROOT/'configs/tiny_aya_stopping_candidate.json').read_text())
    preflight=json.loads((ROOT/'research/results/tiny-aya-corrective-preflight/20261114T213811Z.analysis.json').read_text())
    if (candidate['candidate_eos_token_ids']!=[3,6,261001] or not preflight['all_legacy_controls_reproduced']
        or not preflight['all_candidate_prefix_checks_passed']):
        raise ValueError('stopping preflight/candidate changed')
    prepare_legacy(destination,source)
    path=destination/'configs/tiny_aya_support_rollout.json'
    config=json.loads(path.read_text());before=copy.deepcopy(config)
    historical=json.loads((ROOT/'research/results/tiny-aya-weighted-replay/20261114T161125Z.metadata.json').read_text())['config']
    if before!=historical:raise ValueError('legacy replay configuration drifted from EXP-012')
    if candidate['model_id']!=config['model']['repo_id'] or candidate['model_revision']!=config['model']['revision']:
        raise ValueError('candidate model differs from replay')
    config['config_id']='terminal-slm-native-stops-support-replay-v1'
    config['replay'].update(eos_token_ids=[3,6,261001],capture_stop_audit=True,
        stop_policy_version='upstream-eos-plus-chat-closings-v1',source_stop_preflight_run_id='20261114T213811Z')
    # No instruction, tool, model, task, quantization or existing inference field changes.
    restored=copy.deepcopy(config);restored['config_id']=before['config_id']
    for key in ('eos_token_ids','capture_stop_audit','stop_policy_version','source_stop_preflight_run_id'):
        restored['replay'].pop(key)
    if restored!=before:raise ValueError('unexpected replay configuration difference')
    path.write_text(json.dumps(config,indent=2)+'\n')
    return {'config_sha256':digest(config),'config_id':config['config_id'],'task_ids':config['replay']['task_ids'],
        'adapter_revision':config['replay']['adapter_revision'],'eos_token_ids':[3,6,261001],
        'heldout_accessed':False,'training_performed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--training-result',type=Path)
    args=parser.parse_args();print(json.dumps(prepare(args.destination,args.training_result)))
