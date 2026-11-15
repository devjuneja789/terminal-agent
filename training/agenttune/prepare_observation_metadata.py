"""Audit serialization parity and stage a metadata-only live-history comparison."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from training.agenttune.prepare_observation_intervention import prepare as prepare_source
from training.agenttune.prepare_sft import ROOT,action_rows,digest,read_rows,verified_references

SOURCE=ROOT/'research/results/tiny-aya-observation-intervention/20261114T182159Z.jsonl'


def references() -> list[dict]:
    data=ROOT/'data/sft'
    return verified_references(read_rows(data/'gold_support.jsonl'),read_rows(data/'gold_support.review.jsonl'),read_rows(data/'gold_support.approved.jsonl'))


def audit_format(refs: list[dict]) -> dict:
    from training.agenttune.evaluate_heldout import _bounded_result
    from training.agenttune.hf_gpu_smoke import TinyAyaPolicy
    observations=[]
    for ref in refs:
        for index,message in enumerate(ref['messages']):
            if message['role']!='user' or not message['content'].startswith('Tool result (JSON):\n'):continue
            header,payload=message['content'].split('\n',1)
            value=json.loads(payload)
            normalized=_bounded_result(value,100000)
            observations.append({'task_id':ref['source_task_id'],
                'tool':json.loads(ref['messages'][index-1]['content'])[0]['tool_name'],
                'canonical_pretty_json':message['content']==header+'\n'+json.dumps(value,ensure_ascii=False,indent=2),
                'runtime_serializer_same_bytes_for_same_payload':TinyAyaPolicy._format_observation('Tool result:\n'+repr(value))==message['content'],
                'metadata':value['metadata'],
                'bounded_runtime_changes_only_metadata':normalized=={**value,'metadata':{}},
                'metadata_removed_by_runtime':bool(value['metadata'])})
    return {'observation_count':len(observations),'canonical_pretty_count':sum(r['canonical_pretty_json'] for r in observations),
        'runtime_serializer_identical_count':sum(r['runtime_serializer_same_bytes_for_same_payload'] for r in observations),
        'metadata_removed_count':sum(r['metadata_removed_by_runtime'] for r in observations),
        'observations':observations,'training_performed':False,'heldout_accessed':False}


def metadata_cases(record: dict, metadata: dict) -> list[dict]:
    if record['arm']!='adapter' or record['condition']!='full_source' or record['task_id'] not in ('B021','B026','B041'):
        raise ValueError('expected fixed EXP-014 full-source adapter history')
    messages=record['messages']
    header,payload=messages[-1]['content'].split('\n',1)
    value=json.loads(payload)
    if value['metadata']!={} or header!='Tool result (JSON):' or metadata!={'path':'shop/config.py','line_count':18}:
        raise ValueError('unexpected metadata or observation format')
    if len(value['stdout'].splitlines())!=metadata['line_count']:
        raise ValueError('metadata must describe the unchanged complete source')
    restored=copy.deepcopy(messages)
    restored[-1]['content']=header+'\n'+json.dumps({**value,'metadata':metadata},ensure_ascii=False,indent=2)
    original_rebuilt=header+'\n'+json.dumps(value,ensure_ascii=False,indent=2)
    if original_rebuilt!=messages[-1]['content']:raise ValueError('runtime serialization is not canonical')
    return [{'case_id':record['task_id']+':'+condition,'task_id':record['task_id'],'condition':condition,
        'messages':copy.deepcopy(content),'messages_sha256':digest(content),
        'original_messages_sha256':digest(messages),'identity_control':False,
        'intervention_applied':condition=='reviewed_metadata',
        'original_source_lines':18,'complete_source_lines':18,
        'archived_output_sha256':digest(record['generated_text'])}
        for condition,content in [('runtime',messages),('reviewed_metadata',restored)]]


def build_cases(refs: list[dict]) -> list[dict]:
    rows=action_rows(refs)
    records=[r for r in read_rows(SOURCE) if r['arm']=='adapter' and r['condition']=='full_source']
    if [r['task_id'] for r in records]!=['B021','B026','B041']:raise ValueError('unexpected source cases')
    cases=[]
    for record in records:
        row=next(r for r in rows if r['source_task_id']==record['task_id'] and json.loads(r['completion'][0]['content'])[0]['tool_name']=='write_file')
        metadata=json.loads(row['prompt'][-1]['content'].split('\n',1)[1])['metadata']
        cases.extend(metadata_cases(record,metadata))
    return cases


def prepare(destination: Path) -> dict:
    refs=references()
    static=audit_format(refs)
    if static['observation_count']!=12 or static['runtime_serializer_identical_count']!=12 or not all(r['bounded_runtime_changes_only_metadata'] for r in static['observations']):
        raise ValueError('observation format finding changed')
    prepare_source(destination)
    cases=build_cases(refs)
    config_path=destination/'configs/observation_intervention.json'
    config=json.loads(config_path.read_text())
    config.update(config_id='terminal-slm-observation-metadata-parity-v1',control_condition='runtime',
        source_observation_run_id='20261114T182159Z',case_ids=[c['case_id'] for c in cases],cases_sha256=digest(cases))
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    (destination/'data/cases.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    return {'config_sha256':digest(config),'cases_sha256':digest(cases),'prediction_count':12,'format_audit':static,
        'gold_targets_bundled':False,'heldout_accessed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().destination)))
