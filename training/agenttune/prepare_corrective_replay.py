"""Pin the new corrective adapter in the validated native-stop support replay."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from training.agenttune.prepare_native_stop_replay import prepare as native_prepare
from training.agenttune.prepare_sft import ROOT, digest
from training.agenttune.corrective_training_data import IDS, ROWS_SHA256


def validate_result(result: dict, *, config_path: Path | None=None) -> None:
    config, audit = result['config'], result['training_audit']
    fixed = json.loads((config_path or ROOT/'configs/tiny_aya_corrective_sft.json').read_text())
    if any(config[k] != v for k,v in fixed.items() if k != 'artifact_repo'):
        raise ValueError('training configuration differs from the bounded corrective plan')
    if (result['rows_sha256'] != ROWS_SHA256 or set(result['source_examples']) != IDS
        or result['trajectory_count'] != 4 or result['action_target_count'] != 15 or result['heldout_accessed']
        or audit['global_step'] != fixed['max_steps'] or abs(audit['epoch']-fixed['expected_epochs']) > .001 or audit['skipped_optimizer_steps']
        or not result['adapter_changed'] or not result['adapter_reload_passed']
        or not math.isfinite(result['training_metrics']['train_loss'])
        or not re.fullmatch('[0-9a-f]{40}', result['artifact_commit'])
        or not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', config['artifact_repo'])):
        raise ValueError('incomplete or invalid corrective training result')
    checks = audit['target_mask_checks']
    required = {'prompt_equals_generation_prefix', 'all_prompt_labels_masked', 'contiguous_completion_labels',
        'target_labels_equal_input_ids', 'both_closing_tokens_supervised', 'earlier_closing_tokens_masked',
        'decoded_target_equals_reviewed_action', 'full_template_equals_trainer_ids'}
    if len(checks) != 15 or any(not all(c['label_audit'].get(k) is True for k in required) for c in checks):
        raise ValueError('missing or failed corrective label audit')
    annotations = audit['weight_annotations']
    if len(annotations) != 15 or sum(a['expected_tool']=='write_file' for a in annotations) != 3:
        raise ValueError('missing corrective weight audit')


def prepare(destination: Path, training_result: Path | None = None) -> dict:
    if not training_result:
        raise ValueError('completed corrective training result required')
    result = json.loads(training_result.read_text())
    validate_result(result)
    native_prepare(destination)
    path = destination/'configs/tiny_aya_support_rollout.json'
    config = json.loads(path.read_text())
    config['config_id'] = 'terminal-slm-corrective-native-support-replay-v1'
    config['replay'].update(adapter_repo=result['config']['artifact_repo'], adapter_revision=result['artifact_commit'],
        adapter_subfolder=result['run_id']+'/adapter', source_training_run_id=result['run_id'])
    path.write_text(json.dumps(config,indent=2)+'\n')
    return {'config_sha256':digest(config), 'config_id':config['config_id'], 'task_ids':config['replay']['task_ids'],
        'adapter_revision':result['artifact_commit'], 'eos_token_ids':config['replay']['eos_token_ids'], 'heldout_accessed':False}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--training-result',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.destination,args.training_result)))
