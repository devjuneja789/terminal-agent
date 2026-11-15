"""Stage the fixed tool-choice-only objective over the same 15 admitted rows."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from training.agenttune.prepare_corrective_training import prepare as corrective_prepare
from training.agenttune.prepare_corrective_replay import validate_result
from training.agenttune.prepare_sft import ROOT,digest

PROFILE=ROOT/'configs/tiny_aya_tool_choice_sft.json'


def validate_choice_result(result: dict) -> None:
    validate_result(result,config_path=PROFILE)
    checks=result['training_audit']['tool_choice_objective_self_check']
    if (result['loss']!='tool_choice_only' or not checks['nonchoice_logit_invariance']
        or not checks['only_selected_causal_position_has_gradient'] or checks['model_parameters_updated']):
        raise ValueError('choice-only objective was not validated')
    preflight=checks.get('gradient_preflight',{})
    if (preflight.get('scale')!=result['config']['amp_initial_scale'] or preflight.get('batch_count')!=15
        or preflight.get('optimizer_updates')!=0 or not preflight.get('all_gradients_finite') or not preflight.get('gradients_cleared')):
        raise ValueError('all-batch gradient preflight missing or failed')
    for audit in result['training_audit']['weight_annotations']:
        selected=sorted({p for s in audit['spans'] if s['kind']=='tool_name' for p in s['label_positions']})
        expected=[float(p in selected) for p in audit['label_positions']]
        if (not selected or audit.get('objective')!='tool_choice_only' or audit['selected_tool_positions']!=selected
            or audit['target_weights']!=expected or audit['normalizer']!=len(selected)):
            raise ValueError('nonchoice or missing supervision in tool-only objective')


def prepare(destination: Path,artifact_repo: str,config_path: Path | None=None) -> dict:
    config=json.loads(PROFILE.read_text())
    if config_path and json.loads(config_path.read_text())!=config:
        raise ValueError('choice experiment configuration is fixed')
    manifest=corrective_prepare(destination,artifact_repo)
    config['artifact_repo']=artifact_repo
    (destination/'configs/tiny_aya_sft_smoke.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest.update(config_sha256=digest(config),objective='tool_choice_only',
        objective_scope='tool_name value tokens only; includes finish; zero loss on arguments and closings')
    (destination/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(ROOT/'training/agenttune/tool_choice_objective.py',destination/'training/agenttune/tool_choice_objective.py')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--artifact-repo',required=True)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args();print(json.dumps(prepare(args.destination,args.artifact_repo,args.config)))
