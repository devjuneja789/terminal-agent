"""Freeze one adapter using only the preregistered reviewed-support scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_corrective_contexts import analyze

CONTROL_RESULT=ROOT/'research/results/tiny-aya-joint-sft/20261115T062137Z.result.json'
CONTROL_CONTEXT=ROOT/'research/results/tiny-aya-joint-contexts/20261115T063016Z'


def score(metrics: dict) -> tuple[int,int,int,int]:
    return (metrics['exact_action_matches'],metrics['by_expected_tool']['finish']['tool_name_matches'],
        metrics['schema_valid'],metrics['tool_name_matches'])


def freeze(destination: Path,training_result: Path | None=None,context_prefix: Path | None=None,failure_reason: str | None=None) -> dict:
    if destination.exists():raise FileExistsError('candidate freeze already exists; refusing post-evaluation reselection')
    control=analyze(CONTROL_CONTEXT,CONTROL_RESULT);selected=CONTROL_RESULT;selected_analysis=control
    candidates=[{'training_result':str(CONTROL_RESULT),'context_prefix':str(CONTROL_CONTEXT),
        'score':score(control['metrics']['adapter'])}]
    if training_result is not None and context_prefix is not None:
        result=json.loads(training_result.read_text())
        if result['config']['config_id']!='terminal-slm-joint-fit300-v1':raise ValueError('only fixed fit300 challenger allowed')
        trial=analyze(context_prefix,training_result)
        candidates.append({'training_result':str(training_result),'context_prefix':str(context_prefix),
            'score':score(trial['metrics']['adapter'])})
        if score(trial['metrics']['adapter'])>score(control['metrics']['adapter']):selected=training_result;selected_analysis=trial
    elif not failure_reason:raise ValueError('completed challenger results or a recorded failure reason required')
    source=json.loads(selected.read_text())
    output={'freeze_id':'terminal-slm-submission-candidate-v1',
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'selection_split':'train-support',
        'selection_rule':['exact_full_action_matches','valid_finish_choices','schema_valid_outputs','valid_tool_choices'],
        'tie_break':'retain joint60','candidates':candidates,'failure_reason':failure_reason,
        'selected_training_result':str(selected),'selected_training_result_sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),
        'selected_score':score(selected_analysis['metrics']['adapter']),
        'model_id':source['config']['model_id'],'model_revision':source['config']['model_revision'],
        'adapter_repo':source['config']['artifact_repo'],'adapter_revision':source['artifact_commit'],
        'adapter_subfolder':source['run_id']+'/adapter','rows_sha256':source['rows_sha256'],
        'heldout_used_for_selection':False,'further_training_planned':False}
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'configs/submission_candidate.json')
    parser.add_argument('--training-result',type=Path);parser.add_argument('--context-prefix',type=Path)
    parser.add_argument('--failure-reason')
    args=parser.parse_args();print(json.dumps(freeze(args.output,args.training_result,args.context_prefix,args.failure_reason),indent=2))
