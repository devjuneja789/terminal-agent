"""Execute assistant-authored corrections to actual support histories for review.

No model inference, training, approval or benchmark scoring is performed.
Archived assistant turns are context only; only explicit continuation indices
can become targets after a separate owner review and admission step.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import shlex
import shutil
import tempfile
from pathlib import Path

from evals.benchmark.loader import load_benchmark
from tools import Toolbox
from tools.definitions import validate_tool_arguments
from training.agenttune.build_gold_support import _run_validator,_task_file_hashes
from training.agenttune.evaluate_heldout import _bounded_result
from training.agenttune.prepare_observation_intervention import REPLAY
from training.agenttune.prepare_sft import ROOT,digest,read_rows

SPECS=(
    ('C-B021-read-complete-v1','B021',1,'repair'),
    ('C-B026-choose-write-v1','B026',2,'repair'),
    ('C-B041-preserve-file-v1','B041',1,'repair'),
    ('C-B046-explain-evidence-v1','B046',2,'explanation'),
    ('C-B021-truthful-failure-v1','B021',3,'audit_only_failure_report'),
    ('C-B041-truthful-failure-v1','B041',3,'audit_only_failure_report'),
)


def _action(name: str,**arguments) -> dict:
    action={'tool_name':name,'parameters':arguments}
    if name=='finish':
        if set(arguments)!={'answer'} or not isinstance(arguments['answer'],str):raise ValueError('invalid finish')
    else:validate_tool_arguments(name,arguments)
    return action


def _contract(task) -> dict:
    return {**{k:getattr(task,k) for k in ('task_id','category','split','instruction','expected_outcome','success_condition')},
        'validation_command':list(task.validation_command)}


def _changed(before: dict,after: dict) -> list[str]:
    return sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))


def _evidence_failures(task,final: str,executions: list[dict],validation: dict,changed: list[str]) -> list[str]:
    condition=task.success_condition; failures=[]
    for term in condition.get('response_contains',[]):
        if term.casefold() not in final.casefold():failures.append('missing_response_term:'+term)
    tools={e['tool'] for e in executions}
    failures.extend('missing_tool:'+t for t in condition.get('required_tools',[]) if t not in tools)
    if validation['exit_code'] not in condition.get('validation_exit_codes',[0]):failures.append('validation_failed')
    if condition.get('no_task_file_changes') and changed:failures.append('unexpected_source_change')
    for evidence in condition.get('tool_evidence',[]):
        found=False
        for execution in executions:
            arguments,result=execution['arguments'],execution['result']
            if execution['tool']!=evidence['tool']:continue
            output=result.get('stdout','')+'\n'+result.get('stderr','')
            if (all(str(v).casefold() in arguments.get('command','').casefold() for v in evidence.get('command_contains',[]))
                and all(str(v).casefold() in output.casefold() for v in evidence.get('result_contains',[]))
                and all(arguments.get(k)==v for k,v in evidence.get('arguments_match',{}).items())):
                found=True
        if not found:failures.append('missing_tool_evidence:'+evidence['tool'])
    return failures


def proposed_action_rows(record: dict) -> list[dict]:
    """Inspection-only export; not an approval or training admission function."""
    boundary=record['context_message_count']
    if digest(record['messages'][:boundary])!=record['context_sha256']:
        raise ValueError('archived context changed')
    if record['purpose']=='audit_only_failure_report':return []
    targets=record['proposed_target_message_indices']
    if targets!=[i for i,m in enumerate(record['messages']) if i>=boundary and m['role']=='assistant']:
        raise ValueError('invalid continuation target indices')
    output=[]
    for index in targets:
        message=record['messages'][index]
        action=json.loads(message['content'])
        if len(action)!=1:raise ValueError('expected one native action')
        _action(action[0]['tool_name'],**action[0]['parameters'])
        output.append({'prompt':copy.deepcopy(record['messages'][:index]),'completion':[copy.deepcopy(message)],
            'source_example_id':record['example_id'],'source_task_id':record['source_task_id'],
            'source_message_index':index,'review_required':True,'training_eligible':False})
    return output


def render_review(records: list[dict]) -> str:
    lines=['# Corrective support demonstration review','','All six are assistant-authored continuations with real local tool executions. None is owner-approved or training-eligible. Four task-completing candidates propose 15 action targets; two honest failure reports are audit-only and excluded. No model or training ran.','','Change each review sidecar decision to approved/rejected, add reviewer and reason, and retain the source_record_sha256. Approval does not itself export training data.','','Archived prefix turns are context only. Full exact messages and replay evidence are in the JSONL. Temporary paths in test errors are genuine execution output.','']
    for record in records:
        audit=record['reference_audit'];boundary=record['context_message_count']
        lines += ['## '+record['example_id'],'',f"Task: {record['source_task_id']}; purpose: {record['purpose']}; context messages: {boundary}; proposed targets: {len(record['proposed_target_message_indices'])}.",'',
            '**User instruction:** '+record['messages'][1]['content'],'',
            f"Original-prefix hash: `{record['context_sha256']}`. Admission: `{record['admission_policy']}`.",'',
            f"Pristine/start/final validator exits: {audit['pristine_validation']['exit_code']} / {audit['correction_start_validation']['exit_code']} / {audit['validation']['exit_code']}. Task contract satisfied: {audit['task_contract_satisfied']}.",'',
            '### Last archived tool observation (context only)','','```json',record['messages'][boundary-2]['content'],'```','']
        observation=json.loads(record['messages'][boundary-1]['content'].split('\n',1)[1])
        lines += ['```text',observation.get('stdout','')+observation.get('stderr',''),'```','','### Corrective continuation','']
        for message in record['messages'][boundary:]:
            if message['role']=='assistant':
                action=json.loads(message['content'])[0]
                if action['tool_name']=='write_file':lines += ['write_file: complete config.py; exact full contents are in JSONL and the source diff is below.','']
                else:lines += ['```json',message['content'],'```','']
            else:
                result=json.loads(message['content'].split('\n',1)[1])
                lines += [f"Observed exit_code={result.get('exit_code')}, success={result.get('success')}; runtime metadata={result.get('metadata')}.",'',
                    '```text',result.get('stdout','')+result.get('stderr',''),'```','']
        lines += ['### Source diff from pristine','','```diff',audit['source_diff_from_pristine'] or '(no changes)','```','']
        if record['purpose']=='audit_only_failure_report':lines += ['This is a truthful status report of a still-failed task, not a successful repair. It has zero proposed training targets.','']
    return '\n'.join(lines)


def build(output: Path) -> list[dict]:
    # The candidate, review and packet are an immutable review batch.
    paths=[output,output.with_suffix('.review.jsonl'),output.with_suffix('.review.md'),output.with_suffix('.manifest.json')]
    if any(p.exists() for p in paths):raise FileExistsError('refusing to overwrite an existing review batch')
    tasks={t.task_id:t for t in load_benchmark('train-support')}
    archive={r['task_id']:r for r in read_rows(REPLAY) if r['arm']=='adapter'}
    records=[]
    for example_id,task_id,model_index,purpose in SPECS:
        task=tasks[task_id];source=archive[task_id]
        if source['run_id']!='20261114T161125Z' or source['split']!='train-support' or source['snapshot_sha256']!=task.snapshot_sha256:
            raise ValueError('unexpected source lineage')
        if source['success_condition']!=task.success_condition:
            raise ValueError('task condition differs from the source run')
        if source['agenttune_trace']['model_prompts'][model_index][1]['content']!=task.instruction:
            raise ValueError('source task instruction changed')
        context=copy.deepcopy(source['agenttune_trace']['model_prompts'][model_index])
        messages=copy.deepcopy(context);boundary=len(context)
        limit=source['experiment_metadata']['config']['agent']['max_tool_output_chars']
        executions=[];replayed=[]
        with tempfile.TemporaryDirectory(prefix='terminal-slm-corrective-') as temp:
            workspace=Path(temp)/'workspace';shutil.copytree(task.snapshot_path,workspace)
            original=(workspace/'shop/config.py').read_text()
            before=_task_file_hashes(workspace)
            pristine=_run_validator(task.validation_command,workspace)
            with Toolbox(workspace) as toolbox:
                def execute(action):
                    name,arguments=action['tool_name'],action['parameters']
                    validate_tool_arguments(name,arguments)
                    # Only the recorded named test may execute while rebuilding context.
                    result=_bounded_result(getattr(toolbox,name)(**arguments).as_dict(),limit)
                    if result.get('error') is not None:raise RuntimeError(f'tool could not execute: {result}')
                    return {'tool':name,'arguments':copy.deepcopy(arguments),'result':result}

                for index in range(2,boundary,2):
                    action=json.loads(context[index]['content'])
                    if len(action)!=1:raise ValueError('non-canonical archived action')
                    action=action[0]
                    if action['tool_name']=='shell' and action['parameters']['command']!=shlex.join(task.validation_command):
                        raise ValueError('unexpected shell command in archived prefix')
                    execution=execute(action)
                    recorded=json.loads(context[index+1]['content'].split('\n',1)[1])
                    result=execution['result']
                    if result.get('exit_code')!=recorded.get('exit_code') or result.get('success')!=recorded.get('success'):
                        raise ValueError('replayed context outcome differs from archive')
                    if execution['tool'] in ('read_file','write_file') and result.get('stdout')!=recorded.get('stdout'):
                        raise ValueError('replayed file state differs from archive')
                    replayed.append(execution)
                start_validation=_run_validator(task.validation_command,workspace)
                correction_before=_task_file_hashes(workspace)

                def append_tool(action):
                    messages.append({'role':'assistant','content':json.dumps([action],ensure_ascii=False)})
                    execution=execute(action);executions.append(execution)
                    messages.append({'role':'user','content':'Tool result (JSON):\n'+json.dumps(execution['result'],ensure_ascii=False,indent=2)})
                    return execution['result']

                if purpose=='repair':
                    if task_id in ('B021','B041'):
                        observed=append_tool(_action('read_file',path='shop/config.py',start_line=1,end_line=30))['stdout']
                    else:
                        observed=json.loads(context[-1]['content'].split('\n',1)[1])['stdout']
                    if observed!=original:raise ValueError('complete original source must be observed before a write')
                    if task_id=='B021':
                        old,new='return min(value, MAX_PAGE_SIZE)','return max(1, min(value, MAX_PAGE_SIZE))'
                        final='Fixed page_size() to return at least 1 while retaining MAX_PAGE_SIZE. The named lower-bound test and existing upper-bound test both pass.'
                    elif task_id=='B026':
                        old,new='return DEFAULT_TIMEOUT_SECONDS','return int(os.environ.get(REQUEST_TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS))'
                        final='Updated request_timeout() to read REQUEST_TIMEOUT_SECONDS through REQUEST_TIMEOUT_ENV, with DEFAULT_TIMEOUT_SECONDS as fallback. The previously failing named test now passes; the page-size upper-bound check also passes.'
                    else:
                        old,new='DEFAULT_TIMEOUT_SECONDS = 10','DEFAULT_TIMEOUT_SECONDS = 15'
                        final='Changed DEFAULT_TIMEOUT_SECONDS to 15 while preserving request_timeout() and the rest of the module. The named timeout-default test and page-size upper-bound check both pass.'
                    if observed.count(old)!=1:raise ValueError('ambiguous reference edit')
                    fixed=observed.replace(old,new,1)
                    append_tool(_action('write_file',path='shop/config.py',content=fixed))
                    result=append_tool(_action('shell',command=shlex.join(task.validation_command),timeout=30))
                    if result['exit_code']!=0:raise RuntimeError('corrective named test failed')
                    upper=append_tool(_action('shell',command='python3 -m unittest tests.test_contracts.Contracts.test_page_size_upper_bound',timeout=30))
                    if upper['exit_code']!=0:raise RuntimeError('corrective upper-bound regression check failed')
                elif purpose=='explanation':
                    final='The test fails with AssertionError: 10 != 25. It sets REQUEST_TIMEOUT_SECONDS to 25, but request_timeout() returns DEFAULT_TIMEOUT_SECONDS (10) and does not read the environment variable. No files were changed.'
                elif task_id=='B021':
                    final="The task is not fixed. My write replaced shop/config.py with an incomplete assignment. The named test fails with NameError: name 'page_size' is not defined. I have not produced a passing repair, and the complete original module is not present in these observations."
                else:
                    final="The task is not fixed or verified. My write removed the function definitions from shop/config.py. The named test fails during import with NameError: name 'request_timeout' is not defined. Setting the constant alone did not repair the module; I have no passing test result."
                messages.append({'role':'assistant','content':json.dumps([_action('finish',answer=final)],ensure_ascii=False)})
                validation=_run_validator(task.validation_command,workspace)
                changed=_changed(before,_task_file_hashes(workspace))
                correction_changed=_changed(correction_before,_task_file_hashes(workspace))
                failures=_evidence_failures(task,final,[*replayed,*executions],validation,changed)
                if purpose!='audit_only_failure_report' and failures:raise RuntimeError(f'correction does not satisfy task: {failures}')
                if purpose=='repair' and changed!=['shop/config.py']:raise RuntimeError('repair changed unexpected files')
                if purpose=='audit_only_failure_report' and (validation['exit_code']==0 or executions or correction_changed):
                    raise RuntimeError('audit-only failure report must not claim/perform a repair')
                resulting=(workspace/'shop/config.py').read_text()
            indices=[i for i,m in enumerate(messages) if i>=boundary and m['role']=='assistant']
            record={'schema':'terminal-slm-corrective-review-v1','example_id':example_id,
                'source_task_id':task_id,'source_split':'train-support','snapshot_sha256':task.snapshot_sha256,
                'task_contract_sha256':digest(_contract(task)),'purpose':purpose,
                'authorship':'assistant-authored corrective continuation; not a model success',
                'source_replay':str(REPLAY.relative_to(ROOT)),'source_record_sha256':digest(source),
                'source_run_id':source['run_id'],'source_model_output_index':model_index,
                'context_message_count':boundary,'context_sha256':digest(context),'messages':messages,
                'proposed_target_message_indices':indices if purpose!='audit_only_failure_report' else [],
                'audit_only_assistant_indices':indices if purpose=='audit_only_failure_report' else [],
                'reference_audit':{'context_replay':replayed,'continuation_tool_results':executions,
                    'pristine_validation':pristine,'correction_start_validation':start_validation,'validation':validation,
                    'task_contract_satisfied':not failures,'contract_failures':failures,
                    'changed_files_from_pristine':changed,'changed_files_by_correction':correction_changed,
                    'source_diff_from_pristine':''.join(difflib.unified_diff(original.splitlines(True),resulting.splitlines(True),fromfile='original/shop/config.py',tofile='result/shop/config.py')),
                    'human_review_status':'pending','training_eligible':False},
                'admission_policy':'audit_only_excluded' if purpose=='audit_only_failure_report' else 'owner_review_then_separate_admission',
                'training_eligible':False}
            proposed_action_rows(record)
            records.append(record)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    reviews=[{'example_id':r['example_id'],'source_record_sha256':digest(r),'decision':'pending','reviewer':None,'reason':''} for r in records]
    output.with_suffix('.review.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reviews))
    manifest={'batch_kind':'corrective_support_review_candidates','candidate_count':len(records),
        'task_ids':sorted({r['source_task_id'] for r in records}),'independent_task_count':4,
        'training_candidate_count':4,'audit_only_failure_reports':2,
        'proposed_action_target_count':sum(len(proposed_action_rows(r)) for r in records),
        'training_eligible_count':0,'training_performed':False,'model_inference_performed':False,
        'heldout_accessed':False,'records_sha256':digest(records),'source_replay_sha256':digest(read_rows(REPLAY)),
        'review_required':True,'archived_prefixes_are_not_targets':True}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    output.with_suffix('.review.md').write_text(render_review(records))
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    records=build(parser.parse_args().output)
    print(f'Prepared {len(records)} pending review records; no training admission or inference performed.')
