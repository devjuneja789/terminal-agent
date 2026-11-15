"""Stage only the exact admitted contexts and their completed training audits."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from training.agenttune.prepare_corrective_preflight import verify_export
from training.agenttune.prepare_corrective_replay import validate_result
from training.agenttune.corrective_training_data import verified_rows
from training.agenttune.prepare_sft import ROOT, digest, read_rows

SOURCE=ROOT/'research/results/tiny-aya-corrective-sft/20261114T222003Z.result.json'


def prepare(destination: Path, training_result: Path | None=None) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('bundle directory must be empty')
    source=json.loads((training_result or SOURCE).read_text());validate_result(source)
    if source['artifact_commit']!='3c151e4280bab3fe27c25a53992a3cc17fc785fc':
        raise ValueError('diagnostic is fixed to the EXP-020 corrective adapter')
    rows,_=verify_export()
    refs=read_rows(ROOT/'data/corrective/support_corrective.v1.admitted.jsonl')
    verified_rows(refs,rows)
    stops=json.loads((ROOT/'configs/tiny_aya_native_stopping_v1.json').read_text())
    if stops['status']!='support_replay_validated' or stops['eos_token_ids']!=[3,6,261001]:
        raise ValueError('native stopping profile changed')
    config={'config_id':'terminal-slm-exact-corrective-contexts-v1','source_training_run_id':source['run_id'],
        'adapter_repo':source['config']['artifact_repo'],'adapter_revision':source['artifact_commit'],
        'adapter_subfolder':source['run_id']+'/adapter','model_id':source['config']['model_id'],
        'model_revision':source['config']['model_revision'],'runtime':source['packages'],
        'rows_sha256':digest(rows),'source_sha256':digest(source),'arms':['base','adapter'],
        'context_count':15,'prediction_count':30,'seed':42,'eos_token_ids':stops['eos_token_ids'],
        'max_new_tokens':256,'max_prompt_tokens':7000,'max_context_tokens':8192,
        'max_inference_seconds':1200,'do_sample':False,'use_cache':True,
        'loss_weighting':source['config']['loss_weighting'],'training_performed':False,
        'tool_execution_performed':False,'heldout_accessed':False}
    for relative in ('training/__init__.py','training/agenttune/__init__.py',
        'training/agenttune/prepare_sft.py','training/agenttune/weighted_loss.py',
        'training/agenttune/corrective_training_data.py','training/agenttune/corrective_contexts.py',
        'training/agenttune/run_corrective_contexts_job.sh'):
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    shutil.copytree(ROOT/'tools',destination/'tools',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (destination/'data').mkdir();(destination/'configs').mkdir()
    for name,values in (('rows',rows),('approved',refs)):
        (destination/f'data/{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    (destination/'data/training_result.json').write_text(json.dumps(source,indent=2)+'\n')
    (destination/'configs/corrective_contexts.json').write_text(json.dumps(config,indent=2)+'\n')
    return {'config_sha256':digest(config),'rows_sha256':digest(rows),'adapter_revision':source['artifact_commit'],
        'context_count':15,'prediction_count':30,'training_performed':False,'tool_execution_performed':False,'heldout_accessed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--training-result',type=Path)
    args=parser.parse_args();print(json.dumps(prepare(args.destination,args.training_result)))
