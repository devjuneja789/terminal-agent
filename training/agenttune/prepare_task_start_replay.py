"""Replay the bounded curriculum adapter on its four approved support tasks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from training.agenttune.prepare_native_stop_replay import prepare as native_prepare
from training.agenttune.prepare_task_start_training import validate_result
from training.agenttune.prepare_sft import digest


def prepare(destination: Path,training_result: Path | None=None) -> dict:
    if training_result is None:raise ValueError('completed curriculum result required')
    result=json.loads(training_result.read_text());validate_result(result)
    native_prepare(destination)
    path=destination/'configs/tiny_aya_support_rollout.json';config=json.loads(path.read_text())
    config['config_id']='terminal-slm-task-start-620-support-replay-v1'
    config['replay'].update(adapter_repo=result['config']['artifact_repo'],adapter_revision=result['artifact_commit'],
        adapter_subfolder=result['run_id']+'/adapter',source_training_run_id=result['run_id'])
    path.write_text(json.dumps(config,indent=2)+'\n')
    return {'config_id':config['config_id'],'config_sha256':digest(config),'task_ids':config['replay']['task_ids'],
        'adapter_revision':result['artifact_commit'],'heldout_accessed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--training-result',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.destination,args.training_result)))
