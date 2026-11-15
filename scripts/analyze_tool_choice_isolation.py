"""Compare the isolated tool-only objective with EXP-021; no model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_corrective_contexts import analyze
from training.agenttune.prepare_sft import read_rows


def compare(prefix: Path,training_result: Path) -> dict:
    audit=analyze(prefix,training_result)
    path=prefix.with_suffix('.analysis.json');path.write_text(json.dumps(audit,indent=2)+'\n')
    old_prefix=ROOT/'research/results/tiny-aya-corrective-contexts/20261114T224529Z'
    before={r['record_id']:r for r in read_rows(old_prefix.with_suffix('.prediction.jsonl'))}
    after={r['record_id']:r for r in read_rows(prefix.with_suffix('.prediction.jsonl'))}
    old=json.loads(old_prefix.with_suffix('.summary.json').read_text())
    training=json.loads(training_result.read_text())
    control=json.loads((ROOT/'research/results/tiny-aya-corrective-sft/20261114T222003Z.result.json').read_text())
    changes={k for k in training['config'] if training['config'][k]!=control['config'].get(k)}
    allowed={'config_id','experiment_kind','objective','artifact_repo','amp_initial_scale','gradient_preflight_all_batches'}
    if changes!=allowed or training['rows_sha256']!=control['rows_sha256']:
        raise ValueError('training changed beyond objective and recorded startup correction')
    base_ids=[f'base:{i:02d}' for i in range(15)]
    controls={'contexts':15,
        'exact_generated_token_matches':sum(after[k]['generated_token_ids']==before[k]['generated_token_ids'] for k in base_ids),
        'exact_decoded_output_matches':sum(after[k]['generated_text']==before[k]['generated_text'] for k in base_ids),
        'exact_teacher_top1_matches':sum(after[k]['teacher_forced']['top1_token_ids']==before[k]['teacher_forced']['top1_token_ids'] for k in base_ids)}
    current=audit['metrics']['adapter'];previous=old['metrics']['adapter']
    return {'run_id':audit['run_id'],'training_run_id':training['run_id'],
        'all_independent_checks_passed':True,'training_config_changed_fields':sorted(changes),
        'base_controls':controls,'previous_metrics':previous,'tool_only_metrics':current,
        'primary_memorization_criterion':'15/15 strict-schema-valid correct tool choices',
        'primary_memorization_criterion_met':current['tool_name_matches']==15,
        'tool_choice_delta':current['tool_name_matches']-previous['tool_name_matches'],
        'exact_action_delta':current['exact_action_matches']-previous['exact_action_matches'],
        'wrong_tool_action_indices':[i for i in range(15) if not after[f'adapter:{i:02d}']['comparison']['tool_name_match']],
        'analysis_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'heldout_accessed':False,'tool_execution_performed':False,'full_agent_replay_performed':False,
        'limitations':['Known training contexts, not generalization or complete repairs.',
            'AMP startup scale changed from 8192 to 256 after a zero-update overflow abort; not a perfectly matched numerical control.',
            'Arguments and native closing tokens were not optimized.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',type=Path,required=True);parser.add_argument('--training-result',type=Path,required=True)
    args=parser.parse_args();result=compare(args.prefix,args.training_result)
    path=args.prefix.with_suffix('.isolation.json');path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));print(path)
