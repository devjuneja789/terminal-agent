"""Audit source-supplied predictions; optionally validate only their file contents.

This is a diagnostic, not an agent/benchmark run. No generated shell commands
are executed, and validator passes are never reported as complete task passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.agenttune.prepare_sft import action_rows, digest, read_rows, verified_references
from training.agenttune.source_preservation import inspect_write_prediction


def validate_content(task, content: str, path: str, *, validation_command: tuple[str,...] | None=None) -> dict:
    from evals.benchmark.runner import run_task
    from tools import Toolbox
    if validation_command is not None:
        task=replace(task,validation_command=validation_command)
    with tempfile.TemporaryDirectory(prefix='terminal-slm-preservation-') as temp:
        def apply(_instruction,workspace):
            with Toolbox(workspace) as tools:
                result=tools.write_file(path,content)
                if not result.success:
                    raise RuntimeError(result.error)
            return ''
        run=run_task(task,apply,workspace_parent=Path(temp),validation_timeout=30)
        return {'command':list(task.validation_command),'exit_code':run.validation_exit_code,
            'stdout':run.validation_stdout,'stderr':run.validation_stderr,
            'timed_out':run.validation_timed_out,'apply_error':run.agent_error,
            'changed_files':list(run.changed_files),
            'passed':run.validation_exit_code==0 and not run.validation_timed_out and run.agent_error is None}


def analyze(prefix: Path, output: Path, *, run_validators: bool=False) -> dict:
    data=ROOT/'data/sft'
    refs=verified_references(read_rows(data/'gold_support.jsonl'),read_rows(data/'gold_support.review.jsonl'),read_rows(data/'gold_support.approved.jsonl'))
    rows=action_rows(refs)
    source_meta=json.loads(prefix.with_suffix('.metadata.json').read_text())
    source_summary=json.loads(prefix.with_suffix('.summary.json').read_text())
    if (source_meta['source_training_run_id']!='20261114T160511Z' or
        source_meta['adapter_commit']!='1c0559e7eeab4aaea8fffeddf91ff741cfdb2fe0' or
        source_meta['rows_sha256']!=digest(rows) or source_summary['prediction_count']!=32):
        raise ValueError('unexpected weighted diagnostic lineage')
    old_prefix=ROOT/'research/results/tiny-aya-action-diagnostic/20261114T152447Z'
    old_boundaries={r['action_index']:r for r in read_rows(old_prefix.with_suffix('.boundary.jsonl'))}
    new_boundaries={r['action_index']:r for r in read_rows(prefix.with_suffix('.boundary.jsonl'))}
    if len(new_boundaries)!=16 or any(new_boundaries[i]['input_ids']!=old_boundaries[i]['input_ids'] or new_boundaries[i]['labels']!=old_boundaries[i]['labels'] for i in range(16)):
        raise ValueError('approved diagnostic token sequences changed')
    predictions=read_rows(prefix.with_suffix('.prediction.jsonl'))
    if len(predictions)!=32 or len({r['record_id'] for r in predictions})!=32:
        raise ValueError('expected exactly 32 unique predictions')
    candidates=[]
    for record in predictions:
        if record['expected_tool']=='write_file':
            candidates.append((record['arm'] if record['arm']=='base' else 'weighted64', 'approved_full_source', record))
    for record in read_rows(old_prefix.with_suffix('.prediction.jsonl')):
        if record['arm']=='adapter' and record['expected_tool']=='write_file':
            candidates.append(('unweighted64','approved_full_source',record))
    replay_path=ROOT/'research/results/tiny-aya-weighted-replay/20261114T161125Z.jsonl'
    # First decision after source read: B026 already had the full file in the live run.
    for record in read_rows(replay_path):
        if record['arm']!='adapter' or record['task_id']=='B046':continue
        index=2 if record['task_id']=='B026' else 1
        candidates.append(('weighted64_live','live_full_source' if record['task_id']=='B026' else 'live_partial_source',
            {'source_task_id':record['task_id'],'generated_text':record['agenttune_trace']['model_outputs'][index]}))
    write_rows={r['source_task_id']:r for r in rows if json.loads(r['completion'][0]['content'])[0]['tool_name']=='write_file'}
    tasks={}
    initial_validations={}
    if run_validators:
        from evals.benchmark.loader import load_benchmark
        tasks={t.task_id:t for t in load_benchmark('train-support') if t.task_id in write_rows}
        for key,row in write_rows.items():
            task=tasks[key];ref=next(r for r in refs if r['source_task_id']==key)
            contract={k:getattr(task,k) for k in ('task_id','category','split','instruction','expected_outcome','validation_command','success_condition')}
            # Dataclass validation commands are tuples; the approved JSON used lists.
            contract['validation_command']=list(contract['validation_command'])
            if digest(contract)!=ref['task_contract_sha256'] or task.snapshot_sha256!=ref['snapshot_sha256']:
                raise ValueError('support task contract or snapshot changed')
            score=inspect_write_prediction(row['completion'][0]['content'],row)
            if (task.snapshot_path/score['expected_path']).read_text()!=score['original_content']:
                raise ValueError('approved full source differs from snapshot')
            initial_validations[key]=validate_content(task,score['original_content'],score['expected_path'])
            if initial_validations[key]['passed'] or initial_validations[key]['changed_files']:
                raise ValueError('pristine edit task must fail validation without source changes')
    audits=[]
    for arm,condition,record in candidates:
        row=write_rows[record['source_task_id']]
        score=inspect_write_prediction(record['generated_text'],row)
        audit={'arm':arm,'condition':condition,'task_id':record['source_task_id'],
            'generated_text':record['generated_text'],'training_target_eligible':False,
            **score,'validation_performed':False}
        if run_validators and score['eligible_write']:
            audit['validation']=validate_content(tasks[audit['task_id']],score['generated_content'],score['expected_path'])
            audit['validation_performed']=True
            if audit['task_id']=='B021':
                audit['post_hoc_existing_upper_bound_test']=validate_content(tasks['B021'],score['generated_content'],score['expected_path'],
                    validation_command=('python3','-m','unittest','tests.test_contracts.Contracts.test_page_size_upper_bound'))
        if 'teacher_forced' in record:audit['teacher_forced']=record['teacher_forced']
        audits.append(audit)
    if len(audits)!=12:raise ValueError('expected nine approved-prefix and three live comparisons')
    metrics={}
    for arm in ('base','unweighted64','weighted64','weighted64_live'):
        group=[r for r in audits if r['arm']==arm]
        metrics[arm]={'edit_decisions':len(group),'write_file_selected':sum(r['eligible_write'] for r in group),
            'gold_content_exact':sum(r['gold_content_exact'] for r in group),
            'gold_ast_equal':sum(r['gold_ast_equal'] for r in group),
            'full_file_preserved':sum(r['full_file_preserved'] for r in group),
            'validators_run':sum(r['validation_performed'] for r in group),
            'validators_passed':sum(r.get('validation',{}).get('passed',False) for r in group)}
    summary={'run_kind':'source_preservation_diagnostic','source_diagnostic_run_id':source_meta['run_id'],
        'adapter_commit':source_meta['adapter_commit'],'rows_sha256':digest(rows),'metrics':metrics,
        'pristine_task_validations':initial_validations,
        'same_approved_tokens_and_labels_as_EXP011':True,'training_performed':False,
        'heldout_accessed':False,'training_target_eligible':False,
        'interpretation':'single-action predictions conditioned on approved histories; not agent success or generalization'}
    summary['post_hoc_note']='B021 existing upper-bound test added after inspecting its generated edit; reported separately without changing frozen task validation or benchmark scores.'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.preservation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audits))
    output.with_suffix('.preservation-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    sources=[prefix.with_suffix('.prediction.jsonl'),prefix.with_suffix('.boundary.jsonl'),old_prefix.with_suffix('.prediction.jsonl'),old_prefix.with_suffix('.boundary.jsonl'),replay_path]
    output.with_suffix('.preservation-provenance.json').write_text(json.dumps({str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-prefix',type=Path,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True)
    parser.add_argument('--run-validators',action='store_true')
    args=parser.parse_args()
    print(json.dumps(analyze(args.diagnostic_prefix,args.output_prefix,run_validators=args.run_validators),indent=2))
