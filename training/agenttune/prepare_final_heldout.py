"""Stage a matched held-out comparison only after an immutable support freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from training.agenttune.prepare_hf_eval_bundle import prepare as heldout_prepare
from training.agenttune.prepare_sft import ROOT,digest

HELDOUT_IDS=[f'B{i:03d}' for i in range(5,51,5)]


def prepare(destination: Path,freeze_path: Path | None=None) -> dict:
    frozen=freeze_path or ROOT/'configs/submission_candidate.json'
    candidate=json.loads(frozen.read_text())  # Refuse before touching held-out files.
    if (candidate['freeze_id']!='terminal-slm-submission-candidate-v1'
        or candidate['heldout_used_for_selection'] is not False or candidate['further_training_planned'] is not False):
        raise ValueError('candidate was not frozen from support-only selection')
    source=Path(candidate['selected_training_result'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=candidate['selected_training_result_sha256']:
        raise ValueError('frozen training result changed')
    committed=subprocess.check_output(['git','show','HEAD:evals/benchmark/heldout_eval.jsonl'],cwd=ROOT)
    current=(ROOT/'evals/benchmark/heldout_eval.jsonl').read_bytes()
    if committed!=current:raise ValueError('held-out task bytes changed from committed benchmark')
    ids=list(heldout_prepare(destination))
    if ids!=HELDOUT_IDS:raise ValueError('held-out IDs differ from permanent split')
    runtime=json.loads((ROOT/'research/results/tiny-aya-native-stop-replay/20261114T220329Z.metadata.json').read_text())['config']
    config={key:runtime[key] for key in ('model','runtime','generation','agent','prompt','hashes','prompt_variant','rollout_controls')}
    if candidate['model_id']!=config['model']['repo_id'] or candidate['model_revision']!=config['model']['revision']:
        raise ValueError('frozen base model differs from evaluation runtime')
    config.update(config_id='terminal-slm-final-paired-heldout-v1',evaluation={'benchmark_id':'terminal-slm-initial-v1',
        'split':'held-out-eval','task_ids':ids},heldout_taskfile_sha256=hashlib.sha256(current).hexdigest(),
        candidate_freeze_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        paired={'arms':['base','adapter'],'task_ids':ids,'seed':42,'eos_token_ids':[3,6,261001],
            'adapter_repo':candidate['adapter_repo'],'adapter_revision':candidate['adapter_revision'],
            'adapter_subfolder':candidate['adapter_subfolder'],'extra_runtime':runtime['replay']['extra_runtime'],
            'quantization':runtime['replay']['quantization']})
    (destination/'configs/final_heldout.json').write_text(json.dumps(config,indent=2)+'\n')
    shutil.copy2(frozen,destination/'configs/submission_candidate.json')
    for relative in ('training/agenttune/final_heldout.py','training/agenttune/paired_replay.py',
        'training/agenttune/support_rewards.py','training/agenttune/run_final_heldout_job.sh'):
        shutil.copy2(ROOT/relative,destination/relative)
    if (destination/'data').exists() or (destination/'evals/benchmark/train_support.jsonl').exists():
        raise ValueError('training data in final held-out bundle')
    return {'config_sha256':digest(config),'task_ids':ids,'candidate_freeze_sha256':config['candidate_freeze_sha256'],
        'adapter_revision':candidate['adapter_revision'],'heldout_training_or_reward_generation':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.destination)))
