"""Derive a versioned curriculum exclusively from previously approved sources."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.agenttune.prepare_sft import verified_references,read_rows,action_rows,digest
from training.agenttune.prepare_corrective_preflight import verify_export


def prepare(destination: Path) -> dict:
    refs=verified_references(read_rows(ROOT/'data/sft/gold_support.jsonl'),
        read_rows(ROOT/'data/sft/gold_support.review.jsonl'),read_rows(ROOT/'data/sft/gold_support.approved.jsonl'))
    gold=action_rows(refs);corrections,_=verify_export()
    if len(gold)!=16 or len(corrections)!=15:raise ValueError('approved sources changed')
    rows=[];seen=set()
    for family,sources in [('approved_gold',gold),('admitted_corrective',corrections)]:
        for source in sources:
            key=digest({'prompt':source['prompt'],'completion':source['completion']})
            if key in seen:raise ValueError('unexpected duplicate target')
            seen.add(key)
            rows.append({**source,'source_family':family,'source_split':'train-support',
                'source_export_row_sha256':digest(source),'curriculum_version':'task-start-plus-corrections-v1'})
    first=[r for r in rows if len(r['prompt'])==2]
    if len(first)!=4 or {r['source_task_id'] for r in rows}!={'B021','B026','B041','B046'}:
        raise ValueError('task-start coverage or split changed')
    if destination.exists():raise FileExistsError('versioned curriculum already exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest={'curriculum_version':'task-start-plus-corrections-v1','action_target_count':31,
        'source_task_ids':['B021','B026','B041','B046'],'initial_action_targets':4,
        'gold_rows_sha256':digest(gold),'corrective_rows_sha256':digest(corrections),'rows_sha256':digest(rows),
        'source_reviews_reused':True,'new_model_generated_targets':False,'heldout_accessed':False,
        'audit_only_examples_excluded':True,'historical_failed_prefixes_are_targets':False,
        'training_started':False,'pinned_token_label_preflight':'required_before_training',
        'interpretation':'New mixture of unchanged previously approved targets; not evidence of successful model training.'}
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


if __name__=='__main__':
    print(json.dumps(prepare(ROOT/'data/sft/tiny_aya_task_start_curriculum_v1.jsonl'),indent=2))
