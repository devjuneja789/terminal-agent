"""Verify paired final records against their freeze and recorded evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evals.auditkit_report import recorded_contract_failures
from training.agenttune.evaluate_heldout import _aggregate


def audit(prefix: Path) -> dict:
    records=[json.loads(l) for l in prefix.with_suffix('.jsonl').read_text().splitlines() if l.strip()]
    metadata=json.loads(prefix.with_suffix('.metadata.json').read_text())
    summary=json.loads(prefix.with_suffix('.summary.json').read_text())
    freeze_path=ROOT/'configs/submission_candidate.json';frozen=json.loads(freeze_path.read_text())
    config=metadata['config'];paired=config['paired']
    expected={f'{arm}:B{i:03d}' for arm in ('base','adapter') for i in range(5,51,5)}
    if len(records)!=20 or {r['record_id'] for r in records}!=expected:raise ValueError('incomplete final pairs')
    if metadata['candidate_freeze_sha256']!=hashlib.sha256(freeze_path.read_bytes()).hexdigest():raise ValueError('freeze changed')
    if hashlib.sha256((ROOT/'evals/benchmark/heldout_eval.jsonl').read_bytes()).hexdigest()!=config['heldout_taskfile_sha256']:
        raise ValueError('held-out task bytes changed')
    for key in ('adapter_repo','adapter_revision','adapter_subfolder'):
        if paired[key]!=frozen[key]:raise ValueError('model differs from freeze: '+key)
    if (paired['eos_token_ids']!=[3,6,261001] or metadata['generation_stop_ids']!=[3,6,261001]
        or metadata['training_performed'] or metadata['reward_generation_performed'] or metadata['model_selection_performed']):
        raise ValueError('final protocol changed')
    metrics={};generation_audit={}
    for record in records:
        if (record['run_id']!=summary['run_id'] or record['split']!='held-out-eval'
            or record.get('training_target_eligible') is not False or record.get('training_use_prohibited') is not True
            or 'support_reward' in record):raise ValueError('held-out provenance/training boundary failed')
        if (record['model_id']!=frozen['model_id'] or record['model_revision']!=frozen['model_revision']
            or record['experiment_metadata']['config']!=config):raise ValueError('per-record inference configuration differs')
        if set(recorded_contract_failures(record))!=set(record['criterion_failures']):raise ValueError('contract recomputation differs')
        if record['success']!=(not record['criterion_failures']):raise ValueError('success differs from executable contract')
        if any(g['configured_stop_ids']!=[3,6,261001] or 'decoded_text' not in g for g in record['generation_audit']):
            raise ValueError('generation telemetry incomplete')
    for arm in ('base','adapter'):
        values=[r for r in records if r['arm']==arm]
        metrics[arm]=_aggregate(values)
        if metrics[arm]!=summary['metrics'][arm]:raise ValueError('summary differs from records')
        generations=[g for r in values for g in r['generation_audit']]
        generation_audit[arm]={'captured_generations':len(generations),
            'cap_hits':sum(g['hit_token_cap'] for g in generations),
            'native_stop_hits':sum(g['stopped_on_configured_id'] for g in generations),
            'captured_generated_tokens':sum(g['generated_tokens'] for g in generations),
            'harness_generated_tokens':sum(r['token_usage']['generated_tokens'] for r in values),
            'note':'Captured generations also retain calls rejected after a cooperative deadline; harness token accounting may exclude them.'}
    result={'run_id':summary['run_id'],'all_freeze_contract_and_summary_checks_passed':True,
        'candidate_freeze_sha256':metadata['candidate_freeze_sha256'],'metrics':metrics,
        'generation_audit':generation_audit,'heldout_training_use_prohibited':True,
        'source_records_sha256':hashlib.sha256(prefix.with_suffix('.jsonl').read_bytes()).hexdigest()}
    prefix.with_suffix('.audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prefix',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.prefix),indent=2))
