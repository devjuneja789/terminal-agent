"""Measure the frozen joint-300 adapter from actual support task starts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.agenttune.prepare_native_stop_replay import prepare as native_prepare
from training.agenttune.prepare_joint_training import validate_joint_result
from training.agenttune.prepare_sft import ROOT, digest

SOURCE=ROOT/'research/results/tiny-aya-fit300/20261115T064937Z.result.json'


def prepare(destination: Path,training_result: Path | None=None) -> dict:
    source=training_result or SOURCE
    result=json.loads(source.read_text())
    validate_joint_result(result,config_path=ROOT/'configs/tiny_aya_joint_fit300.json')
    if result['artifact_commit']!='1eed40612640c78d88d9fe0cac36e06b1148d1fd':
        raise ValueError('support diagnostic requires the completed frozen joint-300 artifact')
    native_prepare(destination)
    path=destination/'configs/tiny_aya_support_rollout.json'
    config=json.loads(path.read_text())
    config['config_id']='terminal-slm-joint300-task-start-support-v1'
    config['replay'].update(adapter_repo=result['config']['artifact_repo'],adapter_revision=result['artifact_commit'],
        adapter_subfolder=result['run_id']+'/adapter',source_training_run_id=result['run_id'])
    path.write_text(json.dumps(config,indent=2)+'\n')
    return {'config_id':config['config_id'],'config_sha256':digest(config),
        'task_ids':config['replay']['task_ids'],'adapter_revision':result['artifact_commit'],
        'heldout_accessed':False,'training_performed':False,'new_training_admission':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--training-result',type=Path)
    args=parser.parse_args();print(json.dumps(prepare(args.destination,args.training_result)))
