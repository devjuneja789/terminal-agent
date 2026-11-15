"""Prepare the same 15 exact contexts for the joint-objective adapter."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from training.agenttune.prepare_corrective_contexts import prepare as original_prepare
from training.agenttune.prepare_joint_training import validate_joint_result
from training.agenttune.prepare_sft import ROOT,digest


def prepare(destination: Path,training_result: Path) -> dict:
    source=json.loads(training_result.read_text())
    fit300=source['config']['config_id']=='terminal-slm-joint-fit300-v1'
    validate_joint_result(source,config_path=ROOT/'configs/tiny_aya_joint_fit300.json' if fit300 else None)
    original_prepare(destination)
    path=destination/'configs/corrective_contexts.json';config=json.loads(path.read_text())
    config.update(config_id='terminal-slm-joint-fit300-contexts-v1' if fit300 else 'terminal-slm-joint-choice-contexts-v1',source_training_run_id=source['run_id'],
        adapter_repo=source['config']['artifact_repo'],adapter_revision=source['artifact_commit'],
        adapter_subfolder=source['run_id']+'/adapter',source_sha256=digest(source),
        objective=source['config']['objective'],decision_share=source['config']['decision_share'])
    path.write_text(json.dumps(config,indent=2)+'\n')
    (destination/'data/training_result.json').write_text(json.dumps(source,indent=2)+'\n')
    shutil.copy2(ROOT/'training/agenttune/joint_choice_objective.py',destination/'training/agenttune/joint_choice_objective.py')
    return {'config_sha256':digest(config),'context_count':15,'prediction_count':30,
        'adapter_revision':source['artifact_commit'],'heldout_accessed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--training-result',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.destination,args.training_result)))
