"""Validate matched input controls and inspect predicted file contents locally."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_source_preservation import validate_content
from training.agenttune.prepare_observation_intervention import REPLAY,TRAINING,TASKS,matched_cases
from training.agenttune.prepare_sft import action_rows,digest,read_rows,verified_references
from training.agenttune.source_preservation import inspect_write_prediction
from training.agenttune.weighted_loss import old_file_content


def analyze(prefix: Path, *, run_validators: bool=False) -> dict:
    meta=json.loads(prefix.with_suffix('.metadata.json').read_text())
    config=meta['config']
    training=json.loads(TRAINING.read_text())
    if config['adapter_revision']!=training['artifact_commit'] or config['source_replay_run_id']!='20261114T161125Z':
        raise ValueError('unexpected model/history lineage')
    data=ROOT/'data/sft'
    refs=verified_references(read_rows(data/'gold_support.jsonl'),read_rows(data/'gold_support.review.jsonl'),read_rows(data/'gold_support.approved.jsonl'))
    writes={r['source_task_id']:r for r in action_rows(refs) if json.loads(r['completion'][0]['content'])[0]['tool_name']=='write_file'}
    archived={r['task_id']:r for r in read_rows(REPLAY) if r['arm']=='adapter' and r['task_id'] in TASKS}
    is_metadata=config['config_id']=='terminal-slm-observation-metadata-parity-v1'
    if is_metadata:
        from training.agenttune.prepare_observation_metadata import SOURCE,build_cases
        if config['source_observation_run_id']!='20261114T182159Z':raise ValueError('unexpected metadata control lineage')
        expected_cases=build_cases(refs)
        archived={r['task_id']:r for r in read_rows(SOURCE) if r['arm']=='adapter' and r['condition']=='full_source'}
        conditions=('runtime','reviewed_metadata')
    else:
        expected_cases=[c for task_id in TASKS for c in matched_cases(archived[task_id],old_file_content(writes[task_id],'shop/config.py'))]
        conditions=('original','full_source')
    if digest(expected_cases)!=config['cases_sha256']:raise ValueError('matched inputs differ from archived histories')
    expected={c['case_id']:c for c in expected_cases}
    records=read_rows(prefix.with_suffix('.jsonl'))
    by_id={r['record_id']:r for r in records}
    identities=[arm+':'+c['case_id'] for arm in ('base','adapter') for c in expected_cases]
    if len(records)!=12 or set(by_id)!=set(identities) or meta['record_ids']!=identities:
        raise ValueError('expected twelve unique matched predictions')
    for record in records:
        case=expected[record['case_id']]
        if any(record[k]!=v for k,v in case.items()):raise ValueError('recorded prompt/case changed')
        if record['record_id']!=record['arm']+':'+record['case_id'] or record['run_id']!=meta['run_id'] or record['source_split']!='train-support':
            raise ValueError('prediction identity mismatch')
    for case in expected:
        if by_id['base:'+case]['input_token_ids']!=by_id['adapter:'+case]['input_token_ids']:
            raise ValueError('input token sequences differ between models')
    tasks={};pristine={}
    if run_validators:
        from evals.benchmark.loader import load_benchmark
        tasks={t.task_id:t for t in load_benchmark('train-support') if t.task_id in TASKS}
        for task_id in TASKS:
            task=tasks[task_id];ref=next(r for r in refs if r['source_task_id']==task_id)
            contract={k:getattr(task,k) for k in ('task_id','category','split','instruction','expected_outcome','validation_command','success_condition')}
            contract['validation_command']=list(contract['validation_command'])
            original=old_file_content(writes[task_id],'shop/config.py')
            if digest(contract)!=ref['task_contract_sha256'] or task.snapshot_sha256!=ref['snapshot_sha256'] or (task.snapshot_path/'shop/config.py').read_text()!=original:
                raise ValueError('task contract or snapshot changed')
            pristine[task_id]=validate_content(task,original,'shop/config.py')
            if pristine[task_id]['passed'] or pristine[task_id]['changed_files']:
                raise ValueError('pristine change task must fail without source changes')
    audits=[]
    for record in records:
        score=inspect_write_prediction(record['generated_text'],writes[record['task_id']])
        audit={'record_id':record['record_id'],'task_id':record['task_id'],'arm':record['arm'],
            'condition':record['condition'],'generated_text':record['generated_text'],
            'training_target_eligible':False,'validation_performed':False,**score}
        if run_validators and score['eligible_write']:
            audit['validation']=validate_content(tasks[record['task_id']],score['generated_content'],score['expected_path'])
            audit['validation_performed']=True
            if record['task_id']=='B021':
                audit['existing_upper_bound_test']=validate_content(tasks['B021'],score['generated_content'],score['expected_path'],
                    validation_command=('python3','-m','unittest','tests.test_contracts.Contracts.test_page_size_upper_bound'))
        audits.append(audit)
    lookup={r['record_id']:r for r in audits}
    pairs=[]
    for arm in ('base','adapter'):
        for task_id in TASKS:
            original=by_id[arm+':'+task_id+':'+conditions[0]];full=by_id[arm+':'+task_id+':'+conditions[1]]
            pairs.append({'arm':arm,'task_id':task_id,'identity_control':original['identity_control'],
                'inputs_equal':original['messages']==full['messages'],'input_tokens_equal':original['input_token_ids']==full['input_token_ids'],
                'outputs_equal':original['generated_text']==full['generated_text'],
                'original_selected_tool':lookup[original['record_id']]['action_schema'].get('selected_tool'),
                'full_selected_tool':lookup[full['record_id']]['action_schema'].get('selected_tool')})
    metrics={}
    for arm in ('base','adapter'):
        for condition in conditions:
            group=[r for r in audits if r['arm']==arm and r['condition']==condition]
            raw=[r for r in records if r['arm']==arm and r['condition']==condition]
            metrics[arm+':'+condition]={'decisions':len(group),'write_file_selected':sum(r['eligible_write'] for r in group),
                'full_file_preserved':sum(r['full_file_preserved'] for r in group),'exact_gold_file':sum(r['gold_content_exact'] for r in group),
                'schema_valid':sum(r['action_schema']['schema_valid'] for r in group),
                'validators_run':sum(r['validation_performed'] for r in group),
                'named_validator_passes':sum(r.get('validation',{}).get('passed',False) for r in group),
                'writes_passing_preservation_and_recorded_checks':sum(r['full_file_preserved'] and r.get('validation',{}).get('passed',False) and r.get('existing_upper_bound_test',{'passed':True})['passed'] for r in group),
                'generated_tokens':sum(r['generated_tokens'] for r in raw),'token_cap_hits':sum(r['hit_token_cap'] for r in raw),
                'average_latency_seconds':sum(r['latency_seconds'] for r in raw)/len(raw)}
    controls={task_id:by_id['adapter:'+task_id+':'+conditions[0]]['generated_text']==(archived[task_id]['generated_text'] if is_metadata else archived[task_id]['agenttune_trace']['model_outputs'][2 if task_id=='B026' else 1]) for task_id in TASKS}
    summary={'run_id':meta['run_id'],'run_kind':'matched_observation_metadata_intervention' if is_metadata else 'matched_live_source_observation_intervention','metrics':metrics,'pairs':pairs,
        'control_condition':conditions[0],'intervention_condition':conditions[1],
        'archived_adapter_outputs_reproduced':controls,'all_matched_prompt_checks_passed':True,
        'identity_control_outputs_equal':all(p['outputs_equal'] for p in pairs if p['identity_control']) if any(p['identity_control'] for p in pairs) else None,
        'pristine_validations':pristine,'training_performed':False,'heldout_accessed':False,'training_target_eligible':False,
        'interpretation':'single next-action intervention within fixed archived adapter histories; not complete agent task performance'}
    prefix.with_suffix('.audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audits))
    prefix.with_suffix('.analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    sources=[REPLAY,TRAINING,prefix.with_suffix('.jsonl'),prefix.with_suffix('.metadata.json')]
    if is_metadata:sources.append(SOURCE)
    prefix.with_suffix('.audit-provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--run-validators',action='store_true')
    args=parser.parse_args()
    print(json.dumps(analyze(args.prefix,run_validators=args.run_validators),indent=2))
