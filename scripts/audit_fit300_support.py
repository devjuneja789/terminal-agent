"""Audit actual joint-300 support episodes and rerun recorded file validations."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evals.auditkit_report import recorded_contract_failures
from evals.benchmark.loader import load_benchmark
from scripts.analyze_source_preservation import validate_content
from training.agenttune.evaluate_heldout import _aggregate
from training.agenttune.prepare_sft import digest,read_rows


def audit(prefix: Path,training_result: Path | None=None) -> dict:
    records=read_rows(prefix.with_suffix('.jsonl'))
    metadata=json.loads(prefix.with_suffix('.metadata.json').read_text())
    summary=json.loads(prefix.with_suffix('.summary.json').read_text())
    old=json.loads((ROOT/'research/results/tiny-aya-native-stop-replay/20261114T220329Z.metadata.json').read_text())['config']
    config=metadata['config'];restored=copy.deepcopy(config)
    restored['config_id']=old['config_id']
    for key in ('adapter_repo','adapter_revision','adapter_subfolder','source_training_run_id'):
        restored['replay'][key]=old['replay'][key]
    revision='1eed40612640c78d88d9fe0cac36e06b1148d1fd'
    if training_result is not None:
        from training.agenttune.prepare_task_start_training import validate_result
        training=json.loads(training_result.read_text());validate_result(training)
        revision=training['artifact_commit']
        if (config['config_id']!='terminal-slm-task-start-620-support-replay-v1'
            or config['replay']['adapter_repo']!=training['config']['artifact_repo']
            or config['replay']['adapter_subfolder']!=training['run_id']+'/adapter'):
            raise ValueError('curriculum artifact lineage mismatch')
    if restored!=old or config['replay']['adapter_revision']!=revision:
        raise ValueError('unexpected runtime or adapter drift')
    if digest(config)!=metadata['config_sha256'] or metadata['heldout_accessed'] or summary['heldout_accessed']:
        raise ValueError('config hash or split failure')
    expected={f'{a}:{t}' for a in ('base','adapter') for t in ('B021','B026','B041','B046')}
    if len(records)!=8 or {r['record_id'] for r in records}!=expected:raise ValueError('incomplete pairs')
    tasks={t.task_id:t for t in load_benchmark('train-support')}
    audits=[]
    for record in records:
        if record['split']!='train-support' or record['training_target_eligible'] is not False:
            raise ValueError('split/admission violation')
        if set(recorded_contract_failures(record))!=set(record['criterion_failures']):raise ValueError('contract mismatch')
        executions=record['agenttune_trace']['tool_executions']
        writes=[e for e in executions if e['tool']=='write_file' and isinstance(e['result'],dict) and e['result'].get('success') is True]
        entry={'record_id':record['record_id'],'contract_success':record['success'],
            'successful_write_calls':len(writes),'file_changes':record['task_file_changes'],
            'only_expected_config_changed':set(record['task_file_changes'])=={'shop/config.py'},
            'final_response_source':record['final_response_source'],'final_response':record['final_response']}
        if writes:
            # The diagnostic applies only an actual successful recorded write.
            # Refuse to infer final state from unmodelled shell edits/multiple files.
            if any(e['arguments']['path']!='shop/config.py' for e in writes):
                raise ValueError('additional changed file needs explicit replay audit')
            final=writes[-1]['arguments']
            from training.agenttune.prepare_sft import action_rows
            from training.agenttune.source_preservation import inspect_write_prediction
            gold=next(r for r in action_rows(read_rows(ROOT/'data/sft/gold_support.approved.jsonl'))
                if r['source_task_id']==record['task_id'] and json.loads(r['completion'][0]['content'])[0]['tool_name']=='write_file')
            inspection=inspect_write_prediction(json.dumps([{'tool_name':'write_file','parameters':final}]),gold)
            entry['recorded_write_preservation']={k:v for k,v in inspection.items()
                if k not in ('original_content','gold_content','generated_content','diff_from_original')}
            entry['recorded_write_named_test']=validate_content(tasks[record['task_id']],final['content'],final['path'])
            entry['recorded_write_upper_bound_test']=validate_content(tasks[record['task_id']],final['content'],final['path'],
                validation_command=('python3','-m','unittest','tests.test_contracts.Contracts.test_page_size_upper_bound'))
        audits.append(entry)
    metrics={a:_aggregate([r for r in records if r['arm']==a]) for a in ('base','adapter')}
    if metrics!=summary['metrics']:raise ValueError('aggregate mismatch')
    result={'run_id':summary['run_id'],'runtime_contract_aggregate_checks_passed':True,'metrics':metrics,
        'task_audits':audits,'heldout_accessed':False,'training_admission':False,
        'interpretation':'Known training-support tasks. Re-executed file checks are supplemental diagnostics, not new model episodes.'}
    prefix.with_suffix('.audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--training-result',type=Path)
    args=parser.parse_args();result=audit(args.prefix,args.training_result)
    print(json.dumps(result,indent=2))
