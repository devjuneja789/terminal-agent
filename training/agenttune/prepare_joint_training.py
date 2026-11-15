"""Stage an explicitly balanced native decision/full-action objective."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from training.agenttune.prepare_corrective_training import prepare as corrective_prepare
from training.agenttune.prepare_corrective_replay import validate_result
from training.agenttune.prepare_sft import ROOT,digest
from training.agenttune.joint_choice_objective import joint_weights

PROFILE=ROOT/'configs/tiny_aya_joint_choice_sft.json'


def validate_joint_result(result: dict, *, config_path: Path | None=None) -> None:
    validate_result(result,config_path=config_path or PROFILE)
    check=result['training_audit']['joint_objective_self_check']
    if (result['loss']!='joint_first_choice_completion' or not check['prompt_logits_masked']
        or not check['completion_logits_supervised'] or not check['decision_logits_emphasized']
        or check['model_parameters_updated']):raise ValueError('joint gradient self-check failed')
    preflight=check['gradient_preflight']
    if (preflight['scale']!=256 or preflight['batch_count']!=15 or preflight['optimizer_updates']!=0
        or not preflight['all_gradients_finite'] or not preflight['gradients_cleared']):
        raise ValueError('joint all-batch gradient preflight failed')
    control=json.loads((ROOT/'research/results/tiny-aya-corrective-sft/20261114T222003Z.result.json').read_text())
    original={a['action_index']:a for a in control['training_audit']['weight_annotations']}
    for a in result['training_audit']['weight_annotations']:
        old=original[a['action_index']];labels=[-100]*(max(old['label_positions'])+1);weights=[0.]*len(labels)
        for p,t,w in zip(old['label_positions'],old['target_token_ids'],old['target_weights'],strict=True):labels[p]=t;weights[p]=w
        _,expected=joint_weights(labels,weights,old,result['config']['decision_share'])
        if (a['target_weights']!=expected['target_weights'] or a['decision_position']!=expected['decision_position']
            or not math.isclose(a['normalizer'],1.,abs_tol=1e-12) or any(w<=0 for w in a['target_weights'])):
            raise ValueError('joint coefficients differ from fixed mixture')


def prepare(destination: Path,artifact_repo: str,config_path: Path | None=None) -> dict:
    config=json.loads(PROFILE.read_text())
    if config_path and json.loads(config_path.read_text())!=config:raise ValueError('joint profile is fixed')
    manifest=corrective_prepare(destination,artifact_repo);config['artifact_repo']=artifact_repo
    (destination/'configs/tiny_aya_sft_smoke.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest.update(config_sha256=digest(config),objective=config['objective'],decision_share=config['decision_share'])
    (destination/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for module in ('joint_choice_objective.py','tool_choice_objective.py'):
        shutil.copy2(ROOT/'training/agenttune'/module,destination/'training/agenttune'/module)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--artifact-repo',required=True)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args();print(json.dumps(prepare(args.destination,args.artifact_repo,args.config)))
