"""Apply a hash-bound owner receipt and separately admit corrective continuations.

Raw candidates and the original owner receipt remain immutable. No inference,
training, benchmark modification or audit-only admission occurs here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
import shutil
import tempfile
from pathlib import Path

from evals.benchmark.loader import load_benchmark
from tools import Toolbox
from tools.definitions import validate_tool_arguments
from training.agenttune.build_corrective_support import _contract,_evidence_failures,_changed,proposed_action_rows
from training.agenttune.build_gold_support import _run_validator,_task_file_hashes
from training.agenttune.evaluate_heldout import _bounded_result
from training.agenttune.prepare_observation_intervention import REPLAY
from training.agenttune.prepare_sft import ROOT,SUPPORT_IDS,action_rows,digest,read_rows

ADMIT_IDS=frozenset({'C-B021-read-complete-v1','C-B026-choose-write-v1','C-B041-preserve-file-v1','C-B046-explain-evidence-v1'})
EXCLUDE_IDS=frozenset({'C-B021-truthful-failure-v1','C-B041-truthful-failure-v1'})


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_review(records: list[dict],receipt: dict,reviews: list[dict]) -> dict[str,dict]:
    sources={r['example_id']:r for r in records}
    decisions={r['example_id']:r for r in receipt['decisions']}
    old={r['example_id']:r for r in reviews}
    expected=ADMIT_IDS|EXCLUDE_IDS
    if (len(sources)!=len(records) or len(decisions)!=len(receipt['decisions']) or len(old)!=len(reviews)
        or set(sources)!=expected or set(decisions)!=expected or set(old)!=expected):
        raise ValueError('review must bind exactly the six unique corrective records')
    if receipt['records_sha256']!=digest(records) or receipt['reviewer']!='project_owner':
        raise ValueError('owner receipt does not match candidate batch')
    for key,record in sources.items():
        decision=decisions[key]
        if (decision['source_record_sha256']!=digest(record) or old[key]['source_record_sha256']!=digest(record)
            or decision['decision']!='approved' or old[key]['decision']!='pending'):
            raise ValueError('unapproved, modified or already-applied record')
        if record['source_split']!='train-support' or record['source_task_id'] not in SUPPORT_IDS:
            raise ValueError('only fixed support records may be admitted')
        if key in EXCLUDE_IDS:
            if (record['purpose']!='audit_only_failure_report' or record['admission_policy']!='audit_only_excluded'
                or record['proposed_target_message_indices'] or proposed_action_rows(record)
                or decision['scope']!='factual_accuracy_only_audit_excluded'):
                raise ValueError('audit-only reports are permanently excluded')
        elif (record['purpose'] not in ('repair','explanation') or
              record['admission_policy']!='owner_review_then_separate_admission' or
              decision['scope']!='review_only_admission_pending' or
              not record['reference_audit']['task_contract_satisfied']):
            raise ValueError('candidate is not eligible for separate admission')
    return decisions


def revalidate(record: dict,task,archive: dict) -> dict:
    if (record['task_contract_sha256']!=digest(_contract(task)) or record['snapshot_sha256']!=task.snapshot_sha256
        or record['source_record_sha256']!=digest(archive)):
        raise ValueError('task/snapshot/archive lineage changed')
    context=archive['agenttune_trace']['model_prompts'][record['source_model_output_index']]
    if record['messages'][:record['context_message_count']]!=context or digest(context)!=record['context_sha256']:
        raise ValueError('archived prefix changed')
    proposed_action_rows(record)
    final=json.loads(record['messages'][-1]['content'])[0]
    if final['tool_name']!='finish':raise ValueError('missing reviewed final action')
    limit=archive['experiment_metadata']['config']['agent']['max_tool_output_chars']
    with tempfile.TemporaryDirectory(prefix='terminal-slm-admission-') as temp:
        workspace=Path(temp)/'workspace';shutil.copytree(task.snapshot_path,workspace)
        before=_task_file_hashes(workspace)
        initial=_run_validator(task.validation_command,workspace)
        executions=[]
        with Toolbox(workspace) as toolbox:
            for index in range(2,len(record['messages'])-1,2):
                message,observation=record['messages'][index:index+2]
                if message['role']!='assistant' or observation['role']!='user':raise ValueError('invalid message roles')
                actions=json.loads(message['content'])
                if len(actions)!=1:raise ValueError('expected exactly one native action')
                name,arguments=actions[0]['tool_name'],actions[0]['parameters']
                validate_tool_arguments(name,arguments)
                if name not in ('read_file','write_file','shell'):raise ValueError('unexpected tool in fixed review batch')
                if name=='shell' and arguments['command'] not in (shlex.join(task.validation_command),
                    'python3 -m unittest tests.test_contracts.Contracts.test_page_size_upper_bound'):
                    raise ValueError('unexpected command in reviewed continuation')
                if name in ('read_file','write_file') and arguments['path']!='shop/config.py':
                    raise ValueError('unexpected source path')
                result=_bounded_result(getattr(toolbox,name)(**arguments).as_dict(),limit)
                recorded=json.loads(observation['content'].split('\n',1)[1])
                if (result.get('error') is not None or result.get('exit_code')!=recorded.get('exit_code')
                    or result.get('success')!=recorded.get('success')):
                    raise ValueError('execution outcome differs from reviewed evidence')
                if name in ('read_file','write_file') and result['stdout']!=recorded['stdout']:
                    raise ValueError('source contents differ from reviewed evidence')
                executions.append({'tool':name,'arguments':arguments,'result':result,
                    'context_only':index<record['context_message_count']})
        validation=_run_validator(task.validation_command,workspace)
        changed=_changed(before,_task_file_hashes(workspace))
        failures=_evidence_failures(task,final['parameters']['answer'],executions,validation,changed)
        if failures:raise ValueError('reviewed continuation no longer satisfies task: '+str(failures))
        if record['purpose']=='repair' and changed!=['shop/config.py']:raise ValueError('unexpected repair scope')
        return {'example_id':record['example_id'],'initial_validation':initial,'validation':validation,
            'changed_files':changed,'executions':executions,'contract_failures':failures,'verified':True}


def export_rows(admitted: list[dict]) -> tuple[list[dict],dict]:
    unique={};prompts={};duplicates=[];audit=[]
    for record in admitted:
        if (record['example_id'] not in ADMIT_IDS or record['admission']['status']!='admitted'
            or not record['training_eligible'] or not record['reference_audit']['training_eligible']):
            raise ValueError('unadmitted or excluded record')
        for row in proposed_action_rows(record):
            content={'prompt':row['prompt'],'completion':row['completion']}
            fingerprint=digest(content);prompt_key=digest(row['prompt']);completion_key=digest(row['completion'])
            if prompt_key in prompts and prompts[prompt_key]!=completion_key:raise ValueError('conflicting targets for the same prompt')
            prompts[prompt_key]=completion_key
            if fingerprint in unique:
                duplicates.append({'duplicate':record['example_id'],'retained':unique[fingerprint]['source_example_id']})
                continue
            exported={**row,'review_required':False,'training_eligible':True,
                'source_record_sha256':record['admission']['source_record_sha256'],
                'context_message_count':record['context_message_count'],'example_sha256':fingerprint}
            unique[fingerprint]=exported
            audit.append({'example_sha256':fingerprint,'source_example_id':record['example_id'],
                'source_message_index':row['source_message_index'],'context_message_count':record['context_message_count'],
                'archived_prefix_exact':True,'completion_message_count':1,'supervise_prompt':False,
                'supervise_only_completion':True,'tokenizer_label_audit':'pending'})
    return list(unique.values()),{'removed_duplicates':duplicates,'boundary_checks':audit}


def admit(candidate: Path,receipt_path: Path,export_path: Path) -> dict:
    reviews_path=candidate.with_suffix('.review.jsonl')
    backup=reviews_path.with_name(reviews_path.stem+'.pending.jsonl')
    admitted_path=candidate.with_suffix('.admitted.jsonl')
    admission_path=candidate.with_suffix('.admission.json')
    checks_path=candidate.with_suffix('.revalidation.jsonl')
    boundary_path=export_path.with_suffix('.boundaries.jsonl')
    manifest_path=export_path.with_suffix('.manifest.json')
    destinations=[backup,admitted_path,admission_path,checks_path,export_path,boundary_path,manifest_path]
    if any(p.exists() for p in destinations):raise FileExistsError('refusing to overwrite an admission/export')
    records=read_rows(candidate);reviews=read_rows(reviews_path);receipt=json.loads(receipt_path.read_text())
    if sha_file(reviews_path)!=receipt['review_sidecar_sha256_at_receipt']:
        raise ValueError('pending review sidecar changed after owner receipt')
    decisions=validate_review(records,receipt,reviews)
    tasks={t.task_id:t for t in load_benchmark('train-support')}
    archive={r['task_id']:r for r in read_rows(REPLAY) if r['arm']=='adapter'}
    checks=[revalidate(r,tasks[r['source_task_id']],archive[r['source_task_id']]) for r in records if r['example_id'] in ADMIT_IDS]
    applied=[{**r,'decision':decisions[r['example_id']]['decision'],'reviewer':receipt['reviewer'],
        'reason':decisions[r['example_id']]['reason'],'scope':decisions[r['example_id']]['scope'],
        'receipt_sha256':sha_file(receipt_path)} for r in reviews]
    admitted=[]
    for source in records:
        if source['example_id'] not in ADMIT_IDS:continue
        derived=copy.deepcopy(source)
        derived['schema']='terminal-slm-corrective-admitted-v1'
        derived['training_eligible']=True
        derived['reference_audit'].update(human_review_status='approved',training_eligible=True)
        derived['human_review']=next(r for r in applied if r['example_id']==source['example_id'])
        derived['admission']={'status':'admitted','source_record_sha256':digest(source),
            'owner_receipt_sha256':sha_file(receipt_path),'authorization':'User explicitly requested applying reviews and proceeding with separate admission.',
            'revalidation_passed':True}
        admitted.append(derived)
    rows,audit=export_rows(admitted)
    if len(admitted)!=4 or len(rows)!=15:raise ValueError('unexpected corrective batch size')
    old_rows=action_rows(read_rows(ROOT/'data/sft/gold_support.approved.jsonl'))
    old_hashes={digest({'prompt':r['prompt'],'completion':r['completion']}) for r in old_rows}
    sidecar_text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in applied)
    manifest={'kind':'owner_approved_corrective_sft_admission_v1','admitted_example_ids':[r['example_id'] for r in admitted],
        'task_ids':sorted({r['source_task_id'] for r in admitted}),'admitted_count':4,
        'permanently_excluded_example_ids':sorted(EXCLUDE_IDS),'action_target_count':len(rows),
        'duplicate_count':len(audit['removed_duplicates']),'exact_overlap_with_old_gold':sum(r['example_sha256'] in old_hashes for r in rows),
        'old_gold_merged':False,'rows_sha256':digest(rows),'source_candidates_sha256':digest(records),
        'owner_receipt_sha256':sha_file(receipt_path),'review_sidecar_before_sha256':sha_file(reviews_path),
        'review_sidecar_after_sha256':hashlib.sha256(sidecar_text.encode()).hexdigest(),
        'revalidated_count':len(checks),'archived_prefixes_context_only':True,
        'structural_boundary_check_count':len(audit['boundary_checks']),'tokenizer_label_audit':'pending',
        'training_performed':False,'model_inference_performed':False,'heldout_accessed':False}
    for p in destinations:p.parent.mkdir(parents=True,exist_ok=True)
    backup.write_bytes(reviews_path.read_bytes())
    def write_rows(path,values):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    write_rows(admitted_path,admitted);write_rows(checks_path,checks);write_rows(export_path,rows)
    write_rows(boundary_path,audit['boundary_checks'])
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    admission_path.write_text(json.dumps({**manifest,'removed_duplicates':audit['removed_duplicates'],
        'export_path':str(export_path),'candidate_records_unchanged':True,'owner_receipt_unchanged':True},indent=2)+'\n')
    temporary=reviews_path.with_name(reviews_path.name+'.applying')
    temporary.write_text(sidecar_text);temporary.replace(reviews_path)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(admit(args.candidates,args.receipt,args.output),indent=2))
