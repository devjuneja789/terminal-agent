"""Independently check label boundaries and matched EOS-policy predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.agenttune.corrective_preflight import action_valid,inspect_labels
from training.agenttune.prepare_corrective_preflight import verify_export
from training.agenttune.prepare_sft import digest,read_rows
from training.agenttune.weighted_loss import action_spans


def analyze(prefix: Path) -> dict:
    meta=json.loads(prefix.with_suffix('.metadata.json').read_text())
    summary=json.loads(prefix.with_suffix('.summary.json').read_text())
    labels=read_rows(prefix.with_suffix('.label.jsonl'));generations=read_rows(prefix.with_suffix('.generation.jsonl'))
    rows,_=verify_export();config=meta['config']
    if digest(rows)!=config['rows_sha256'] or len(labels)!=15 or len(generations)!=12:
        raise ValueError('source data or preflight counts changed')
    if set(meta['record_ids'])!={r['record_id'] for r in labels+generations} or len(meta['record_ids'])!=27:
        raise ValueError('missing or duplicate records')
    by_source={(r['source_example_id'],r['source_message_index']):r for r in rows}
    for record in labels:
        row=by_source[(record['source_example_id'],record['source_message_index'])]
        audit=inspect_labels(record['input_ids'],record['labels'],record['generation_prompt_ids'],meta['closing_ids'])
        if any(record['audit'][k]!=v for k,v in audit.items()) or not all(v for v in audit.values() if isinstance(v,bool)):
            raise ValueError('label boundary mismatch')
        if record['gold_action']!=row['completion'][0]['content']:raise ValueError('reviewed target changed')
        if not all(v for v in record['audit'].values() if isinstance(v,bool)):raise ValueError('failed label audit')
        if audit['sequence_tokens']>config['max_length']:raise ValueError('truncated or oversized target')
        weights=record['weight_audit']
        expected=action_spans(row,config['loss_weighting'])
        if expected!=[{k:v for k,v in s.items() if k!='label_positions'} for s in weights['spans']]:
            raise ValueError('weighting spans differ from reviewed text')
        if weights['target_weights'][-2:]!=[1,1] or len(weights['target_weights'])!=audit['target_tokens']:
            raise ValueError('closing weights or target lengths changed')
    lookup={r['record_id']:r for r in generations};pairs=[]
    for arm in config['arms']:
        for task in ('B021','B026','B041'):
            old=lookup[f'generation:{arm}:legacy_turn_only:{task}'];new=lookup[f'generation:{arm}:response_or_turn:{task}']
            if old['messages']!=new['messages'] or old['input_token_ids']!=new['input_token_ids']:
                raise ValueError('EOS comparison prompts changed')
            for record in (old,new):
                if record['action']!=action_valid(record['generated_text']):raise ValueError('schema score mismatch')
                if record['stop_ids']!=config['stop_policies'][record['policy']]:raise ValueError('stop policy changed')
            ids=old['generated_token_ids'];first=next((i for i,v in enumerate(ids) if v in new['stop_ids']),None)
            prefix_match=first is not None and new['generated_token_ids']==ids[:first+1]
            pairs.append({'arm':arm,'task_id':task,'legacy_control_reproduced':old['matches_archived_legacy_output'],
                'same_generation_prefix_through_first_closing':prefix_match,
                'legacy_tokens':len(ids),'candidate_tokens':len(new['generated_token_ids']),
                'legacy_cap_hit':old['hit_token_cap'],'candidate_cap_hit':new['hit_token_cap'],
                'decoded_action_unchanged':old['generated_text']==new['generated_text'],
                'candidate_last_token_id':new['last_token_id']})
    metrics={}
    for arm in config['arms']:
        for policy in config['stop_policies']:
            group=[r for r in generations if r['arm']==arm and r['policy']==policy]
            metrics[arm+':'+policy]={'cases':len(group),'generated_tokens':sum(r['generated_tokens'] for r in group),
                'token_cap_hits':sum(r['hit_token_cap'] for r in group),'schema_valid':sum(r['action']['schema_valid'] for r in group),
                'response_end_tokens':sum(r['response_end_count'] for r in group),
                'average_latency_seconds':sum(r['latency_seconds'] for r in group)/len(group)}
    result={'run_id':meta['run_id'],'rows_sha256':config['rows_sha256'],'label_count':len(labels),
        'all_15_label_boundaries_passed':True,'all_15_weight_annotations_passed':True,
        'max_sequence_tokens':max(r['audit']['sequence_tokens'] for r in labels),
        'upstream_pinned_configs':meta['upstream_pinned_configs'],'stop_pairs':pairs,'metrics':metrics,
        'all_legacy_controls_reproduced':all(p['legacy_control_reproduced'] for p in pairs),
        'all_candidate_prefix_checks_passed':all(p['same_generation_prefix_through_first_closing'] for p in pairs),
        'training_performed':False,'heldout_accessed':False,'frozen_inference_settings_changed':False,
        'interpretation':'TRL label audit and fixed-prompt EOS comparison; not a live agent benchmark or model improvement'}
    if summary['label_count']!=15 or summary['generation_count']!=12 or summary['optimizer_created']:
        raise ValueError('unexpected preflight execution')
    prefix.with_suffix('.analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    provenance=json.loads(prefix.with_suffix('.provenance.json').read_text())
    provenance['source_files_sha256']={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in
        ('training/agenttune/corrective_preflight.py','training/agenttune/prepare_corrective_preflight.py',
         'training/agenttune/weighted_loss.py','scripts/analyze_corrective_preflight.py')}
    prefix.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prefix',type=Path,required=True)
    print(json.dumps(analyze(parser.parse_args().prefix),indent=2))
