"""Stage matched live-history observations, without gold answers or training code."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from training.agenttune.prepare_sft import ROOT, action_rows, digest, read_rows, verified_references
from training.agenttune.prepare_replay_bundle import validate_memorization_result
from training.agenttune.weighted_loss import old_file_content

REPLAY=ROOT/'research/results/tiny-aya-weighted-replay/20261114T161125Z.jsonl'
TRAINING=ROOT/'research/results/tiny-aya-weighted-sft/20261114T160511Z.result.json'
TASKS=('B021','B026','B041')


def matched_cases(record: dict, source: str) -> list[dict]:
    if record['arm']!='adapter' or record['split']!='train-support' or record['task_id'] not in TASKS:
        raise ValueError('only the three fixed weighted support histories are allowed')
    index=2 if record['task_id']=='B026' else 1
    messages=record['agenttune_trace']['model_prompts'][index]
    action=json.loads(messages[-2]['content'])[0]
    if action['tool_name']!='read_file' or action['parameters']['path']!='shop/config.py' or action['parameters']['start_line']!=1:
        raise ValueError('history is not at the expected first source read')
    header,payload=messages[-1]['content'].split('\n',1)
    observation=json.loads(payload)
    if header!='Tool result (JSON):' or observation.get('success') is not True:
        raise ValueError('expected a successful JSON source observation')
    if header+'\n'+json.dumps(observation,ensure_ascii=False,indent=2)!=messages[-1]['content']:
        raise ValueError('observation serialization differs from recorded live format')
    expected=''.join(source.splitlines(True)[:action['parameters']['end_line']])
    if observation['stdout']!=expected:
        raise ValueError('recorded source slice does not match the pristine full source')
    full=copy.deepcopy(messages)
    supplied={**observation,'stdout':source}
    full[-1]['content']=header+'\n'+json.dumps(supplied,ensure_ascii=False,indent=2)
    restored=copy.deepcopy(full)
    restored[-1]['content']=messages[-1]['content']
    if restored!=messages:
        raise ValueError('intervention changed earlier conversation messages')
    identity=full==messages
    if identity!=(record['task_id']=='B026'):
        raise ValueError('unexpected identity-control status')
    return [{'case_id':record['task_id']+':'+condition,'task_id':record['task_id'],
        'condition':condition,'messages':copy.deepcopy(value),'messages_sha256':digest(value),
        'original_messages_sha256':digest(messages),'identity_control':identity,
        'intervention_applied':condition=='full_source' and not identity,
        'original_source_lines':len(observation['stdout'].splitlines()),'complete_source_lines':len(source.splitlines()),
        'archived_output_sha256':digest(record['agenttune_trace']['model_outputs'][index])}
        for condition,value in [('original',messages),('full_source',full)]]


def prepare(destination: Path) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('bundle directory must be empty')
    data=ROOT/'data/sft'
    refs=verified_references(read_rows(data/'gold_support.jsonl'),read_rows(data/'gold_support.review.jsonl'),read_rows(data/'gold_support.approved.jsonl'))
    training=json.loads(TRAINING.read_text())
    validate_memorization_result(training,refs)
    if training['artifact_commit']!='1c0559e7eeab4aaea8fffeddf91ff741cfdb2fe0':
        raise ValueError('unexpected weighted adapter revision')
    replay=[r for r in read_rows(REPLAY) if r['arm']=='adapter' and r['task_id'] in TASKS]
    if len(replay)!=3 or {r['task_id'] for r in replay}!=set(TASKS):
        raise ValueError('incomplete archived live histories')
    rows=action_rows(refs)
    cases=[]
    for record in replay:
        if record['run_id']!='20261114T161125Z' or record['experiment_metadata']['config']['replay']['adapter_revision']!=training['artifact_commit']:
            raise ValueError('unexpected live history lineage')
        row=next(r for r in rows if r['source_task_id']==record['task_id'] and json.loads(r['completion'][0]['content'])[0]['tool_name']=='write_file')
        source=old_file_content(row,'shop/config.py')
        cases.extend(matched_cases(record,source))
    generation=replay[0]['experiment_metadata']['config']['generation']
    if any(r['experiment_metadata']['config']['generation']!=generation for r in replay):
        raise ValueError('live generation configurations differ')
    config={'config_id':'terminal-slm-live-observation-intervention-v1',
        'model_id':training['config']['model_id'],'model_revision':training['config']['model_revision'],
        'adapter_repo':training['config']['artifact_repo'],'adapter_revision':training['artifact_commit'],
        'adapter_subfolder':training['run_id']+'/adapter','runtime':training['packages'],
        'seed':42,'generation':generation,'max_inference_seconds':1200,
        'arms':['base','adapter'],'case_ids':[c['case_id'] for c in cases],
        'cases_sha256':digest(cases),'training_performed':False,'heldout_accessed':False,
        'tool_execution_performed':False,'source_replay_run_id':'20261114T161125Z'}
    for relative in ['training/__init__.py','training/agenttune/__init__.py','training/agenttune/prepare_sft.py',
                     'training/agenttune/observation_intervention.py','training/agenttune/run_observation_intervention_job.sh']:
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    (destination/'data').mkdir()
    (destination/'configs').mkdir()
    (destination/'data/cases.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    (destination/'configs/observation_intervention.json').write_text(json.dumps(config,indent=2)+'\n')
    return {'config_sha256':digest(config),'cases_sha256':digest(cases),'case_count':len(cases),
        'prediction_count':len(cases)*2,'heldout_accessed':False,'gold_targets_bundled':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().destination)))
