"""Stage admitted corrective rows and pinned inference-only preflight inputs."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from training.agenttune.admit_corrective_support import ADMIT_IDS,EXCLUDE_IDS,export_rows,sha_file
from training.agenttune.prepare_sft import ROOT,digest,read_rows


def verify_export() -> tuple[list[dict],dict]:
    base=ROOT/'data/corrective/support_corrective.v1.jsonl'
    admission=json.loads(base.with_suffix('.admission.json').read_text())
    source=read_rows(base);by_id={r['example_id']:r for r in source}
    receipt_path=base.with_suffix('.owner-review-receipt.json')
    receipt=json.loads(receipt_path.read_text());reviews=read_rows(base.with_suffix('.review.jsonl'))
    if (digest(source)!=admission['source_candidates_sha256'] or sha_file(receipt_path)!=admission['owner_receipt_sha256']
        or sha_file(base.with_suffix('.review.jsonl'))!=admission['review_sidecar_after_sha256']):
        raise ValueError('admission/review source hashes changed')
    decisions={r['example_id']:r for r in receipt['decisions']}
    for review in reviews:
        if review['decision']!='approved' or review['source_record_sha256']!=digest(by_id[review['example_id']]):
            raise ValueError('review no longer binds its source')
    admitted=read_rows(base.with_suffix('.admitted.jsonl'))
    if {r['example_id'] for r in admitted}!=ADMIT_IDS or len(admitted)!=4:raise ValueError('unexpected admitted set')
    for record in admitted:
        raw=by_id[record['example_id']];review=next(r for r in reviews if r['example_id']==record['example_id'])
        if decisions[record['example_id']]['scope']!='review_only_admission_pending':raise ValueError('wrong review scope')
        expected=copy.deepcopy(raw)
        expected['schema']='terminal-slm-corrective-admitted-v1';expected['training_eligible']=True
        expected['reference_audit'].update(human_review_status='approved',training_eligible=True)
        expected['human_review']=review;expected['admission']=record['admission']
        if record!=expected or record['admission']['source_record_sha256']!=digest(raw) or record['admission']['owner_receipt_sha256']!=sha_file(receipt_path):
            raise ValueError('admitted transcript differs from reviewed source')
    expected,_=export_rows(admitted)
    path=ROOT/'data/sft/tiny_aya_corrective_v1.jsonl';rows=read_rows(path)
    if rows!=expected or digest(rows)!=admission['rows_sha256'] or len(rows)!=15:
        raise ValueError('corrective export changed')
    if any(r['source_example_id'] in EXCLUDE_IDS or not r['training_eligible'] for r in rows):raise ValueError('excluded target')
    return rows,admission


def prepare(destination: Path) -> dict:
    if destination.exists() and any(destination.iterdir()):raise ValueError('bundle directory must be empty')
    rows,admission=verify_export()
    training=json.loads((ROOT/'research/results/tiny-aya-weighted-sft/20261114T160511Z.result.json').read_text())
    replay=read_rows(ROOT/'research/results/tiny-aya-observation-intervention/20261114T182159Z.jsonl')
    cases=[]
    for task_id in ('B021','B026','B041'):
        pair={r['arm']:r for r in replay if r['task_id']==task_id and r['condition']=='full_source'}
        if set(pair)!={'base','adapter'} or pair['base']['messages']!=pair['adapter']['messages']:raise ValueError('source pair changed')
        cases.append({'task_id':task_id,'messages':pair['adapter']['messages'],
            'messages_sha256':digest(pair['adapter']['messages']),
            'legacy_output_sha256':{k:digest(v['generated_text']) for k,v in pair.items()}})
    config={'config_id':'terminal-slm-corrective-label-stop-preflight-v1',
        'model_id':training['config']['model_id'],'model_revision':training['config']['model_revision'],
        'adapter_repo':training['config']['artifact_repo'],'adapter_revision':training['artifact_commit'],
        'adapter_subfolder':training['run_id']+'/adapter','runtime':training['packages'],'seed':42,
        'max_length':4096,'max_new_tokens':256,'max_inference_seconds':1200,
        'arms':['base','adapter'],'stop_policies':{'legacy_turn_only':[6],'response_or_turn':[261001,6]},
        'rows_sha256':digest(rows),'cases_sha256':digest(cases),'label_count':15,'generation_count':12,
        'loss_weighting':training['config']['loss_weighting'],'admission_rows_sha256':admission['rows_sha256'],
        'training_performed':False,'heldout_accessed':False,'source_run_id':'20261114T182159Z'}
    for relative in ['training/__init__.py','training/agenttune/__init__.py','training/agenttune/prepare_sft.py',
                     'training/agenttune/weighted_loss.py','training/agenttune/corrective_preflight.py',
                     'training/agenttune/run_corrective_preflight_job.sh']:
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
    shutil.copytree(ROOT/'tools',destination/'tools',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (destination/'data').mkdir();(destination/'configs').mkdir()
    (destination/'data/rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (destination/'data/cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases))
    (destination/'configs/corrective_preflight.json').write_text(json.dumps(config,indent=2)+'\n')
    return {'config_sha256':digest(config),'rows_sha256':digest(rows),'label_count':15,'generation_count':12,
        'training_performed':False,'heldout_accessed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().destination)))
