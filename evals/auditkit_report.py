"""Offline AuditKIT evaluation of real recorded Terminal-SLM agent episodes.

No model, terminal command, workspace validator or external judge is run here.
Executable validation evidence comes from the original isolated benchmark run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from tools.definitions import get_tool_schemas

AUDITKIT_COMMIT='9d4e21bac240e8108a22c2c137d179d6070868d5'


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def recorded_contract_failures(record: dict) -> list[str]:
    """Recompute frozen runner conditions from observations, not success/reward."""
    condition=record['success_condition'];response=record['final_response'].casefold()
    validation=record['validation'];events=record['trajectory']['events']
    tools={e.get('tool_name') for e in events if e.get('kind')=='tool' and e.get('executed') and e.get('tool_name')}
    required=set(condition.get('required_tools',[]))
    failures=[f'missing_response_term:{str(t).casefold()}' for t in condition.get('response_contains',[]) if str(t).casefold() not in response]
    failures.extend(f'missing_required_tool:{name}' for name in sorted(required-tools))
    output=(validation['stdout']+'\n'+validation['stderr']).casefold()
    failures.extend(f'missing_validation_output:{t}' for t in condition.get('validation_output_contains',[]) if str(t).casefold() not in output)
    tool_output='\n'.join(str(e.get('tool_result',{}).get(k,'')) for e in events
        if e.get('kind')=='tool' and isinstance(e.get('tool_result'),dict) for k in ('stdout','stderr')).casefold()
    failures.extend(f'missing_tool_output:{t}' for t in condition.get('tool_output_contains',[]) if str(t).casefold() not in tool_output)
    for evidence in condition.get('tool_evidence',[]):
        matched=False
        for event in events:
            if event.get('kind')!='tool' or event.get('tool_name')!=evidence['tool'] or event.get('executed') is False:continue
            args=event.get('tool_arguments',{});expected=evidence.get('arguments_match',{})
            if not isinstance(expected,dict) or any(args.get(k)!=v for k,v in expected.items()):continue
            result=event.get('tool_result')
            if isinstance(result,dict) and isinstance(result.get('output'),dict):result=result['output']
            text='\n'.join(str(result.get(k,'')) for k in ('stdout','stderr')).casefold() if isinstance(result,dict) else str(result).casefold()
            command=str(args.get('command','')).casefold()
            if (all(str(t).casefold() in command for t in evidence.get('command_contains',[]))
                and all(str(t).casefold() in text for t in evidence.get('result_contains',[]))):matched=True;break
        if not matched:failures.append(f'missing_tool_evidence:{evidence["tool"]}')
    if validation['exit_code'] not in condition.get('validation_exit_codes',[0]):
        failures.append('validation_timeout' if validation['timed_out'] else 'validation_command_failed')
    if record['agent_stop_reason']!='completed':failures.append('agent_incomplete')
    if condition.get('no_task_file_changes'):
        changes=record.get('task_file_changes',record.get('support_audit',{}).get('changed_files'))
        if changes is None:raise ValueError('file-change evidence missing; cannot verify no-change contract')
        if changes:failures.append('unexpected_task_file_changes')
    return failures


def evaluate_records(records: list[dict], *, name: str='terminal-slm-recorded'):
    from auditkit.agent_eval import AgentCase,AgentEvalRunner,AgentEvalSpec,Oracle,Outcome,episode_from_eventlog

    class RecordedContractOracle(Oracle):
        type='terminal_slm_recorded_contract';version='1'
        def __init__(self,record):self.record=record
        def identity(self):return {'type':self.type,'version':self.version,'record_sha256':digest(self.record)}
        def evaluate(self,episode,case=None):
            try:
                failures=recorded_contract_failures(self.record)
                if set(failures)!=set(self.record['criterion_failures']):
                    raise ValueError('independent contract recomputation differs from recorded runner')
            except (ValueError,KeyError,TypeError) as exc:
                return Outcome('error',reason=str(exc),source=self.identity())
            return Outcome('failure' if failures else 'success',reason='Frozen contract recomputed from recorded executable evidence',
                evidence={'criterion_failures':failures,'validation_exit_code':self.record['validation']['exit_code'],
                    'validation_stdout_stderr_sha256':digest([self.record['validation']['stdout'],self.record['validation']['stderr']])},source=self.identity())

    finish={'type':'function','function':{'name':'finish','description':'Model final response; not an environment tool.',
        'parameters':{'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False}}}
    cases=[];episodes=[];seen=set()
    for record in records:
        identity=record.get('record_id',record['task_id'])
        if identity in seen:raise ValueError('duplicate recorded episode identity')
        seen.add(identity)
        episode=episode_from_eventlog({'events':record['agenttune_trace']['events']},source_id=identity)
        episode.case_id=identity;episode.final_answer=record['final_response'] if record.get('final_response_source')=='model' else None
        episode.coverage['final_answer']='observed' if episode.final_answer is not None else 'unavailable'
        episode.stop_reason=record['agent_stop_reason'];episode.source_verdict=record['success']
        episode.metadata.update(source_record_sha256=digest(record),split=record['split'],
            task_id=record['task_id'],failure_category=record['failure_category'],
            original_agent_error=record.get('agent_error'))
        episode.counters.update(latency_ms=record['latency_seconds']*1000,tokens=record['token_usage']['total_tokens'])
        cases.append(AgentCase(id=identity,task=record['trajectory']['user_request'],category=record['category'],
            allowed_tools=get_tool_schemas()+[finish],outcome=RecordedContractOracle(record)))
        episodes.append(episode)
    result=AgentEvalRunner().run(AgentEvalSpec(cases=cases,episodes=episodes,mode='recorded',
        scorers=['tool_call_validity','redundant_tool_calls','agent_loop_detection'],name=name))
    if any(r.status=='method_error' or r.outcome['verdict']=='error' for r in result.rows):
        raise ValueError('AuditKIT bridge/metric error; do not publish incomplete scores: '+result.summary())
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records',type=Path);parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args();records=[json.loads(l) for l in args.records.read_text().splitlines() if l.strip()]
    distribution=importlib.metadata.distribution('auditkit')
    direct=json.loads(distribution.read_text('direct_url.json') or '{}')
    actual_commit=direct.get('vcs_info',{}).get('commit_id')
    if distribution.version!='1.1.12' or actual_commit!=AUDITKIT_COMMIT:
        raise ValueError('install the actual pinned AuditKIT core from requirements-audit.txt')
    output=args.output_prefix;output.parent.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for arm in sorted({r.get('arm','model') for r in records}):
        values=[r for r in records if r.get('arm','model')==arm]
        result=evaluate_records(values,name=f'terminal-slm-recorded-{arm}-{digest(values)}')
        result.to_json(str(output.with_name(output.name+f'.{arm}.auditkit.json')))
        summaries+=['## '+arm,'',result.summary(),'']
    intro=['# AuditKIT recorded-episode evaluation','','Real imported AgentTune events; no new inference or tool execution.',
        'The headline recomputes the frozen benchmark contract from recorded executable validator evidence. It does not trust model claims or source rewards.',
        'Tool validity/loop metrics describe emitted EventLog actions, including finish; rejected model JSON and pre-emission blocked repeats remain in the separate harness attempt metrics.',
        'Missing scorer evidence is ineligible, not a fabricated zero. Keyword-based answer contracts retain the benchmark’s limitations.','']
    output.with_suffix('.md').write_text('\n'.join(intro+summaries))
    metadata={'auditkit_version':distribution.version,'auditkit_commit':actual_commit,
        'source_records':str(args.records),'source_sha256':hashlib.sha256(args.records.read_bytes()).hexdigest(),
        'bridge_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'episodes':len(records),'mode':'recorded','local_model_inference':False,'tool_execution':False,'external_judge':False}
    output.with_suffix('.provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('\n'.join(summaries));return 0


if __name__=='__main__':raise SystemExit(main())
