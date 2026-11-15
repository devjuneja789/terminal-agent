"""Prepare a paired replay containing exactly the four approved support tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import re
import math
from pathlib import Path

from training.agenttune.prepare_hf_support_bundle import prepare as prepare_support
from training.agenttune.prepare_sft import ROOT, SUPPORT_IDS, action_rows, digest, read_rows, verified_references

TASK_IDS = ('B021', 'B026', 'B041', 'B046')
ADAPTER_REVISION = '3335b1636ba319486ad01d974a37b33506169d8c'


def validate_memorization_result(result: dict, refs: list[dict]) -> None:
    config, audit = result['config'], result['training_audit']
    profiles = {'support_memorization_diagnostic':'tiny_aya_memorization.json',
                'support_weighted_objective':'tiny_aya_weighted_sft.json'}
    if (config.get('experiment_kind') not in profiles or
        config['max_steps'] != 64 or audit['global_step'] != 64 or audit['epoch'] < 3.999 or
        audit['skipped_optimizer_steps'] or not result['adapter_changed'] or not result['adapter_reload_passed']):
        raise ValueError('memorization training did not complete the fixed 64-step plan')
    if result['rows_sha256'] != digest(action_rows(refs)) or result['source_examples'] != [r['example_id'] for r in refs]:
        raise ValueError('training data does not match approved references')
    expected = json.loads((ROOT/'configs'/profiles[config['experiment_kind']]).read_text())
    if any(config[key] != value for key, value in expected.items() if key != 'artifact_repo'):
        raise ValueError('training config differs from the fixed memorization plan')
    if not math.isfinite(result['training_metrics']['train_loss']) or not re.fullmatch(r'[0-9a-f]{40}', result['artifact_commit']):
        raise ValueError('invalid training loss or artifact revision')
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', config['artifact_repo']):
        raise ValueError('invalid adapter repository')
    if config['experiment_kind']=='support_weighted_objective':
        annotations = audit.get('weight_annotations',[])
        if len(annotations)!=16 or sum(a['expected_tool']=='write_file' for a in annotations)!=3:
            raise ValueError('incomplete weighted-target audit')


def prepare(destination: Path, training_result: Path | None = None) -> dict:
    data = ROOT/'data/sft'
    refs = verified_references(read_rows(data/'gold_support.jsonl'), read_rows(data/'gold_support.review.jsonl'), read_rows(data/'gold_support.approved.jsonl'))
    result = json.loads(training_result.read_text()) if training_result else None
    if result:
        validate_memorization_result(result, refs)
    if {r['source_task_id'] for r in refs} != set(TASK_IDS) or set(TASK_IDS) != SUPPORT_IDS:
        raise ValueError('paired replay must use exactly the approved support tasks')
    prepare_support(destination, TASK_IDS)
    task_file = destination/'evals/benchmark/train_support.jsonl'
    tasks = [r for r in read_rows(task_file) if r['task_id'] in TASK_IDS]
    for task in tasks:
        ref = next(r for r in refs if r['source_task_id'] == task['task_id'])
        contract = {key: task[key] for key in ('task_id', 'category', 'split', 'instruction',
            'expected_outcome', 'validation_command', 'success_condition')}
        if digest(contract) != ref['task_contract_sha256'] or task['snapshot_sha256'] != ref['snapshot_sha256']:
            raise ValueError('task changed after gold approval')
    task_file.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in tasks))
    manifest_path = destination/'evals/benchmark/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(task_count=4, splits={'train-support': list(TASK_IDS)})
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    config_path = destination/'configs/tiny_aya_support_rollout.json'
    config = json.loads(config_path.read_text())
    config.update(config_id='terminal-slm-paired-support-replay-v2-template-check',
        support_taskset_sha256=hashlib.sha256(task_file.read_bytes()).hexdigest(),
        replay={'arms': ['base', 'adapter'], 'task_ids': list(TASK_IDS),
                'seed': 42, 'training_examples_replayed': True,
                'observation_format': 'normalized JSON; includes v5 formatting correction',
                'adapter_repo': 'devjuneja789/terminal-slm-sft-smoke-v1',
                'adapter_revision': ADAPTER_REVISION,
                'adapter_subfolder': '20261113T220511Z/adapter',
                'source_training_run_id': '20261113T220511Z',
                'quantization': {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
                    'bnb_4bit_use_double_quant': True, 'bnb_4bit_compute_dtype': 'float16'},
                'extra_runtime': {'peft': '0.21.3', 'bitsandbytes': '0.50.2',
                    'datasets': '5.2.0', 'trl': '1.7.1'}})
    if result:
        config['config_id'] = 'terminal-slm-weighted-support-replay-v1' if result['config']['experiment_kind']=='support_weighted_objective' else 'terminal-slm-memorization-support-replay-v1'
        config['replay'].update(adapter_repo=result['config']['artifact_repo'],
            adapter_revision=result['artifact_commit'], adapter_subfolder=result['run_id']+'/adapter',
            source_training_run_id=result['run_id'])
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    for relative in ['training/agenttune/paired_replay.py', 'training/agenttune/run_replay_job.sh']:
        shutil.copy2(ROOT/relative, destination/relative)
    # No training references/targets or held-out task records enter inference.
    return {'config_sha256': digest(config), 'task_ids': list(TASK_IDS), 'adapter_revision': config['replay']['adapter_revision']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--training-result', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination, args.training_result)))
