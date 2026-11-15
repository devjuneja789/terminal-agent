"""Verify the unchanged approved gold + corrective curriculum in host/job bundles."""
from __future__ import annotations
from pathlib import Path
from training.agenttune.prepare_sft import ROOT,action_rows,digest,read_rows,verified_references
from training.agenttune.corrective_training_data import verified_rows as verify_corrective

ROWS_SHA256='c78909c5b951303ac8b3c0904008eb207017c1a37fb9c6359c55782d0d47f233'


def verified_rows(refs: list[dict],rows: list[dict],proof_root: Path | None=None) -> list[dict]:
    proof=proof_root or ROOT/'data/task_start_proofs'
    gold=verified_references(read_rows(proof/'gold_source.jsonl'),read_rows(proof/'gold_reviews.jsonl'),refs[:4])
    corrections=verify_corrective(refs[4:],read_rows(proof/'corrective_rows.jsonl'))
    expected=[]
    for family,sources in [('approved_gold',action_rows(gold)),('admitted_corrective',corrections)]:
        for source in sources:
            expected.append({**source,'source_family':family,'source_split':'train-support',
                'source_export_row_sha256':digest(source),'curriculum_version':'task-start-plus-corrections-v1'})
    if len(refs)!=8 or len(rows)!=31 or rows!=expected or digest(rows)!=ROWS_SHA256:
        raise ValueError('mixed curriculum differs from approved sources')
    if sum(len(r['prompt'])==2 for r in rows)!=4:
        raise ValueError('initial action targets missing')
    return rows
