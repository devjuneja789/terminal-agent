"""Independently audit exact-context lineage, action scores and native stopping."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.agenttune.corrective_contexts import aggregate,compare_action
from training.agenttune.prepare_corrective_preflight import verify_export
from training.agenttune.prepare_corrective_contexts import SOURCE
from training.agenttune.prepare_corrective_replay import validate_result
from training.agenttune.prepare_sft import digest,read_rows


def require(value: bool,message: str) -> None:
    if not value:raise ValueError(message)


def analyze(prefix: Path,training_result: Path | None=None) -> dict:
    contexts=read_rows(prefix.with_suffix('.context.jsonl'))
    predictions=read_rows(prefix.with_suffix('.prediction.jsonl'))
    metadata=json.loads(prefix.with_suffix('.metadata.json').read_text())
    summary=json.loads(prefix.with_suffix('.summary.json').read_text())
    source_path=training_result or SOURCE
    source=json.loads(source_path.read_text())
    choice_only=source['config'].get('objective')=='tool_choice_only'
    joint=source['config'].get('objective')=='joint_first_choice_completion'
    fit300=source['config']['config_id']=='terminal-slm-joint-fit300-v1'
    if joint:
        from training.agenttune.prepare_joint_training import validate_joint_result
        validate_joint_result(source,config_path=ROOT/'configs/tiny_aya_joint_fit300.json' if fit300 else None)
    elif choice_only:
        from training.agenttune.prepare_tool_choice_training import validate_choice_result
        validate_choice_result(source)
    else:validate_result(source)
    rows,_=verify_export();config=metadata['config']
    expected_id='terminal-slm-joint-choice-contexts-v1' if joint else ('terminal-slm-tool-choice-contexts-v1' if choice_only else 'terminal-slm-exact-corrective-contexts-v1')
    if fit300:expected_id='terminal-slm-joint-fit300-contexts-v1'
    require(config['config_id']==expected_id and digest(config)==metadata['config_sha256'],'config mismatch')
    if choice_only or joint:
        import copy
        control=json.loads((ROOT/'research/results/tiny-aya-corrective-contexts/20261114T224529Z.metadata.json').read_text())['config']
        restored=copy.deepcopy(config);restored.pop('objective')
        if joint:restored.pop('decision_share')
        for key in ('config_id','source_training_run_id','adapter_repo','adapter_revision','adapter_subfolder','source_sha256'):
            restored[key]=control[key]
        require(restored==control,'inference procedure differs from EXP-021 beyond adapter/lineage/objective audit')
    require(config['rows_sha256']==digest(rows) and config['source_sha256']==digest(source),'source hash mismatch')
    require(config['adapter_revision']==source['artifact_commit'] and config['adapter_repo']==source['config']['artifact_repo']
        and config['adapter_subfolder']==source['run_id']+'/adapter','adapter lineage mismatch')
    require(config['eos_token_ids']==[3,6,261001] and config['max_new_tokens']==256 and config['seed']==42
        and config['do_sample'] is False and config['use_cache'] is True,'inference settings mismatch')
    require(config['model_id']==source['config']['model_id'] and config['model_revision']==source['config']['model_revision'],'base model changed')
    require(len(contexts)==15 and len(predictions)==30 and summary['context_count']==15 and summary['prediction_count']==30,'incomplete results')
    for value in (metadata,summary,config):
        require(value['training_performed'] is False and value['tool_execution_performed'] is False and value['heldout_accessed'] is False,'scope violation')
    by_context={c['action_index']:c for c in contexts}
    require(set(by_context)==set(range(15)),'context identity mismatch')
    annotations={a['action_index']:a for a in source['training_audit']['weight_annotations']}
    for index,row in enumerate(rows):
        c=by_context[index];prompt=c['generation_prompt_ids'];ids=c['input_ids'];labels=c['labels'];audit=c['weight_audit'];old=annotations[index]
        require(c['record_id']==f'context:{index:02d}' and c['source_split']=='train-support','context split/id mismatch')
        require(c['messages']==row['prompt'] and c['gold_action']==row['completion'][0]['content'],'reviewed text changed')
        require(c['source_example_id']==row['source_example_id'] and c['source_message_index']==row['source_message_index'],'source index mismatch')
        require(ids[:len(prompt)]==prompt and labels==[-100]*len(prompt)+ids[len(prompt):] and labels[-2:]==[261001,6],'label boundary mismatch')
        for key in ('target_token_ids','label_positions','target_weights'):
            require(audit[key]==old[key],'training-token/label/weight reproduction mismatch')
        require([labels[i] for i in audit['label_positions']]==audit['target_token_ids'],'stored target labels mismatch')
        require(c['training_tokens_labels_weights_reproduced'] is True,'failed training reproduction')
    require({r['record_id'] for r in predictions}=={f'{arm}:{i:02d}' for arm in ('base','adapter') for i in range(15)},'prediction identity mismatch')
    per_action=[]
    for r in predictions:
        c=by_context[r['action_index']];gold=c['gold_action'];ids=r['generated_token_ids']
        require(r['record_id']==f"{r['arm']}:{r['action_index']:02d}" and r['context_record_id']==c['record_id'],'prediction/context link mismatch')
        require(r['source_split']=='train-support' and r['training_target_eligible'] is False,'unexpected training admission')
        require(r['gold_action']==gold and r['expected_tool']==json.loads(gold)[0]['tool_name'],'target changed')
        require(r['comparison']==compare_action(r['generated_text'],gold),'action comparison mismatch')
        closed=bool(ids and ids[-1] in config['eos_token_ids'])
        require(r['configured_stop_ids']==[3,6,261001] and 0<len(ids)<=256,'stop configuration or token limit changed')
        require(r['hit_token_cap']==(len(ids)==256) and (closed or len(ids)==256),'unexpected generation termination')
        require(sum(ids.count(i) for i in [3,6,261001])==int(closed),'tokens emitted after a recognized closing')
        t=r['teacher_forced'];targets=c['weight_audit']['target_token_ids'];positions=c['weight_audit']['label_positions']
        require(t['gold_token_ids']==targets and len(t['top1_token_ids'])==len(targets),'teacher target mismatch')
        correct=[a==b for a,b in zip(targets,t['top1_token_ids'],strict=True)]
        require(t['exact_sequence']==all(correct) and t['token_accuracy']==sum(correct)/len(correct),'teacher accuracy mismatch')
        for kind,score in t['scopes'].items():
            selected={p for s in c['weight_audit']['spans'] if s['kind']==kind for p in s['label_positions']}
            values=[correct[i] for i,p in enumerate(positions) if p in selected]
            require(score=={'tokens':len(values),'top1_correct':sum(values),'exact':all(values) if values else None},'critical-token score mismatch')
        per_action.append({'arm':r['arm'],'action_index':r['action_index'],'task_id':r['source_task_id'],
            'expected_tool':r['expected_tool'],'comparison':r['comparison'],'generated_text':r['generated_text'],
            'teacher_token_accuracy':t['token_accuracy'],'teacher_mean_nll':t['mean_nll'],'teacher_critical_scopes':t['scopes'],
            'hit_token_cap':r['hit_token_cap']})
    metrics=aggregate(predictions)
    require(metrics==summary['metrics'],'aggregate mismatch')
    return {'run_id':summary['run_id'],'config_sha256':metadata['config_sha256'],'all_lineage_token_action_and_stop_checks_passed':True,
        'metrics':metrics,'per_action':sorted(per_action,key=lambda r:(r['action_index'],r['arm'])),
        'heldout_accessed':False,'training_performed':False,'tool_execution_performed':False,
        'interpretation':'Conditional predictions on known training contexts, not agent task success or generalization.',
        'artifact_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source_path,*[prefix.with_suffix(s) for s in
            ('.context.jsonl','.prediction.jsonl','.metadata.json','.summary.json')]]}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prefix',type=Path,required=True)
    parser.add_argument('--training-result',type=Path)
    args=parser.parse_args();result=analyze(args.prefix,args.training_result)
    path=args.prefix.with_suffix('.analysis.json');path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_checks_passed':True,'metrics':result['metrics']}));print(path)
