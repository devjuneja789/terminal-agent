"""Audit the bounded corrective run and its paired support replay, without inference."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.agenttune.prepare_corrective_replay import validate_result
from training.agenttune.evaluate_heldout import _aggregate
from training.agenttune.prepare_sft import digest, read_rows

REFERENCE=ROOT/'research/results/tiny-aya-native-stop-replay/20261114T220329Z'


def require(value: bool, message: str) -> None:
    if not value: raise ValueError(message)


def analyze(training_path: Path, prefix: Path) -> dict:
    training=json.loads(training_path.read_text());validate_result(training)
    records=read_rows(prefix.with_suffix('.jsonl'))
    metadata=json.loads(prefix.with_suffix('.metadata.json').read_text())
    summary=json.loads(prefix.with_suffix('.summary.json').read_text())
    reference=json.loads(REFERENCE.with_suffix('.metadata.json').read_text())
    old_records=read_rows(REFERENCE.with_suffix('.jsonl'))
    config=metadata['config']; restored=copy.deepcopy(config)
    restored['config_id']=reference['config']['config_id']
    for k in ('adapter_repo','adapter_revision','adapter_subfolder','source_training_run_id'):
        restored['replay'][k]=reference['config']['replay'][k]
    require(restored==reference['config'],'replay changed beyond version and adapter identity')
    require(config['config_id']=='terminal-slm-corrective-native-support-replay-v1','wrong replay version')
    require(metadata['config_sha256']==digest(config),'configuration hash mismatch')
    require(config['replay']['adapter_revision']==training['artifact_commit']
        and config['replay']['adapter_repo']==training['config']['artifact_repo']
        and config['replay']['adapter_subfolder']==training['run_id']+'/adapter','adapter lineage mismatch')
    require(metadata['generation_stop_ids']==[3,6,261001] and metadata['generation_stop_audit_enabled'],'incorrect native-stop policy')
    require(metadata['quantization']==reference['quantization'] and metadata['tokenizer_equivalence']==reference['tokenizer_equivalence'],'inference format drift')
    expected=[r['record_id'] for r in old_records]
    require(len(records)==8 and [r['record_id'] for r in records]==expected==summary['record_ids']==metadata['record_ids'],'incomplete paired replay')
    require(not metadata['heldout_accessed'] and not summary['heldout_accessed'],'held-out access')
    audits=[]
    for r, old in zip(records,old_records,strict=True):
        trace=r['agenttune_trace']
        for k in ('task_id','split','model_id','model_revision','snapshot_sha256','success_condition'):
            require(r[k]==old[k],f"{r['record_id']}: {k} drift")
        require(r['split']=='train-support' and r['training_target_eligible'] is False,'unexpected training admission')
        require(r['validation']['command']==old['validation']['command'],'validation command changed')
        require(trace['model_prompts'][0]==old['agenttune_trace']['model_prompts'][0],'initial prompt drift')
        generations=r['generation_audit'];endings=Counter()
        require(len(generations)==len(trace['model_outputs'])>0,'incomplete generation telemetry')
        require(sum(g['generated_tokens'] for g in generations)==r['token_usage']['generated_tokens'],'token accounting mismatch')
        for g in generations:
            ids=g['generated_token_ids']; closed=bool(ids and ids[-1] in [3,6,261001]);cap=config['generation']['max_new_tokens']
            require(g['configured_stop_ids']==[3,6,261001] and 0<len(ids)<=cap,'stop policy or token limit mismatch')
            require(len(ids)==g['generated_tokens'] and ids[-1]==g['last_token_id'],'token audit mismatch')
            require(g['stopped_on_configured_id']==closed and g['hit_token_cap']==(len(ids)==cap),'termination flag mismatch')
            require(closed or len(ids)==cap,'unexplained generation stop')
            require([ids.count(3),ids.count(6),ids.count(261001)]==[g['upstream_eos_count'],g['turn_end_count'],g['response_end_count']],'closing count mismatch')
            require(sum(ids.count(i) for i in [3,6,261001])==int(closed),'generation continued past native closing')
            endings[str(ids[-1]) if closed else 'length_cap']+=1
        writes=[e for e in trace['tool_executions'] if e.get('tool')=='write_file']
        source_checks=[]
        for write in writes:
            arguments=write['arguments']; content=arguments.get('content','')
            check={'path':arguments.get('path'),'source_sha256':hashlib.sha256(content.encode()).hexdigest(),
                'content_characters':len(content),'execution_success':write.get('result',{}).get('success')}
            if arguments.get('path')=='shop/config.py':
                try:
                    module=ast.parse(content)
                    functions=sorted(n.name for n in module.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)))
                    check.update(syntax_valid=True,module_function_names=functions,
                        missing_original_functions=sorted({'request_timeout','page_size'}-set(functions)))
                except SyntaxError as exc:check.update(syntax_valid=False,syntax_error=str(exc))
            source_checks.append(check)
        audits.append({'record_id':r['record_id'],'contract_success':r['success'],'failure_category':r['failure_category'],
            'criterion_failures':r['criterion_failures'],'generation_endings':dict(endings),
            'write_file_executions':len(writes),'changed_files':r['support_audit']['changed_files'],
            'write_source_checks':source_checks,
            'validation_exit_code':r['validation']['exit_code'],'final_response':r['final_response'],
            'final_response_source':r['final_response_source'],
            'selected_actions_equal_reference':trace['selected_actions']==old['agenttune_trace']['selected_actions']})
    metrics={arm:_aggregate([r for r in records if r['arm']==arm]) for arm in ('base','adapter')}
    require(metrics==summary['metrics'],'aggregate metrics mismatch')
    losses=training['training_audit']['log_history']
    epoch_losses={}
    for epoch in range(1,5):
        rows=[r['loss'] for r in losses if 'loss' in r and epoch-1<r['epoch']<=epoch]
        require(len(rows)==15,'incomplete per-epoch training history')
        epoch_losses[str(epoch)]=sum(rows)/len(rows)
    result={'training_run_id':training['run_id'],'replay_run_id':summary['run_id'],
        'training_and_runtime_audits_passed':True,'heldout_accessed':False,
        'training_artifact_commit':training['artifact_commit'],'training_artifact_repo':training['config']['artifact_repo'],
        'epoch_mean_training_loss':epoch_losses,'metrics':metrics,'task_audits':audits,
        'previous_native_metrics':{arm:_aggregate([r for r in old_records if r['arm']==arm]) for arm in ('base','adapter')},
        'generated_tokens':{arm:sum(r['token_usage']['generated_tokens'] for r in records if r['arm']==arm) for arm in ('base','adapter')},
        'interpretation':'Support-only training/replay diagnostic; factual and preservation audit required, not held-out generalization.'}
    result['artifact_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [training_path,prefix.with_suffix('.jsonl'),prefix.with_suffix('.metadata.json'),prefix.with_suffix('.summary.json')]}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-result',type=Path,required=True);parser.add_argument('--prefix',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.training_result,args.prefix)
    path=args.prefix.with_suffix('.analysis.json');path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'training_and_runtime_audits_passed':True,'metrics':result['metrics'],'epoch_mean_training_loss':result['epoch_mean_training_loss']}))
    print(path)
