"""Stage only approved references for an inference-only action/label audit."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from training.agenttune.prepare_sft import ROOT, read_rows, verified_references, digest
from training.agenttune.prepare_sft_bundle import prepare as prepare_sources
from training.agenttune.prepare_replay_bundle import validate_memorization_result


def prepare(destination: Path, training_result: Path) -> dict:
    source = json.loads(training_result.read_text())
    data = ROOT/'data/sft'
    refs = verified_references(read_rows(data/'gold_support.jsonl'),read_rows(data/'gold_support.review.jsonl'),read_rows(data/'gold_support.approved.jsonl'))
    validate_memorization_result(source,refs)
    profile = 'tiny_aya_weighted_sft.json' if source['config']['experiment_kind']=='support_weighted_objective' else 'tiny_aya_memorization.json'
    prepare_sources(destination,source['config']['artifact_repo'],ROOT/'configs'/profile)
    (destination/'training/agenttune/run_sft_job.sh').unlink()
    (destination/'data/training_result.json').write_text(json.dumps(source,indent=2)+'\n')
    config = {'config_id':'terminal-slm-weighted-source-diagnostic-v1' if profile=='tiny_aya_weighted_sft.json' else 'terminal-slm-approved-action-diagnostic-v1',
        'source_training_run_id':source['run_id'], 'adapter_repo':source['config']['artifact_repo'],
        'adapter_commit':source['artifact_commit'], 'action_count':16, 'arms':['base','adapter'],
        'max_new_tokens':256, 'max_inference_seconds':1200,
        'runtime':source['packages'], 'training_performed':False,'tool_execution_performed':False,
        'heldout_accessed':False,'rows_sha256':source['rows_sha256']}
    (destination/'configs/action_diagnostic.json').write_text(json.dumps(config,indent=2)+'\n')
    for relative in ['training/agenttune/action_diagnostic.py','training/agenttune/run_action_diagnostic_job.sh']:
        shutil.copy2(ROOT/relative,destination/relative)
    if (destination/'evals').exists():
        raise ValueError('no benchmark tasks are permitted in this diagnostic')
    return {'config_sha256':digest(config),'adapter_commit':source['artifact_commit'],'action_count':16}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--training-result',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.destination,args.training_result)))
