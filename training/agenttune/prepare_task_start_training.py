"""Stage one fixed 620-update mixed approved-curriculum run."""
from __future__ import annotations
import argparse
import json
import math
import re
import shutil
from pathlib import Path
from training.agenttune.prepare_joint_training import prepare as joint_prepare
from training.agenttune.prepare_sft import ROOT,read_rows,digest
from training.agenttune.task_start_data import verified_rows,ROWS_SHA256

PROFILE=ROOT/'configs/tiny_aya_task_start_sft.json'


def prepare(destination: Path,artifact_repo: str,config_path: Path | None=None) -> dict:
    config=json.loads(PROFILE.read_text())
    if config_path and json.loads(config_path.read_text())!=config:raise ValueError('fixed curriculum profile changed')
    manifest=joint_prepare(destination,artifact_repo)
    proof=destination/'data/task_start_proofs';proof.mkdir()
    for source,name in [('data/sft/gold_support.jsonl','gold_source.jsonl'),
        ('data/sft/gold_support.review.jsonl','gold_reviews.jsonl'),
        ('data/sft/tiny_aya_corrective_v1.jsonl','corrective_rows.jsonl')]:
        shutil.copy2(ROOT/source,proof/name)
    refs=read_rows(ROOT/'data/sft/gold_support.approved.jsonl')+read_rows(ROOT/'data/corrective/support_corrective.v1.admitted.jsonl')
    rows=verified_rows(refs,read_rows(ROOT/'data/sft/tiny_aya_task_start_curriculum_v1.jsonl'),proof)
    for name,values in [('approved',refs),('corrective_rows',rows)]:
        (destination/f'data/{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    config['artifact_repo']=artifact_repo
    (destination/'configs/tiny_aya_sft_smoke.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest.update(config_sha256=digest(config),refs_sha256=digest(refs),rows_sha256=digest(rows),
        source_example_ids=[r['example_id'] for r in refs],action_target_count=31,old_gold_merged=True,
        new_label_gate='all_31_actual_trainer_masks_and_gradients_required_before_updates')
    # The old 15-row label gate cannot certify the new 31-row mixture.
    manifest.pop('label_gate',None)
    (destination/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(ROOT/'training/agenttune/task_start_data.py',destination/'training/agenttune/task_start_data.py')
    return manifest


def validate_result(result: dict) -> None:
    config=result['config'];fixed=json.loads(PROFILE.read_text());audit=result['training_audit']
    if any(config[k]!=v for k,v in fixed.items() if k!='artifact_repo'):raise ValueError('training profile drift')
    expected_sources={f'G-{t}-v1' for t in ('B021','B026','B041','B046')} | {
        'C-B021-read-complete-v1','C-B026-choose-write-v1','C-B041-preserve-file-v1','C-B046-explain-evidence-v1'}
    if set(result['source_examples'])!=expected_sources or len(result['source_examples'])!=8:
        raise ValueError('training source identities changed')
    if (result['rows_sha256']!=ROWS_SHA256 or result['trajectory_count']!=8 or result['action_target_count']!=31
        or result['heldout_accessed'] or audit['global_step']!=620 or abs(audit['epoch']-20)>.001
        or audit['skipped_optimizer_steps'] or not result['adapter_changed'] or not result['adapter_reload_passed']
        or not math.isfinite(result['training_metrics']['train_loss'])
        or not re.fullmatch('[0-9a-f]{40}',result['artifact_commit'])):raise ValueError('training incomplete')
    required=('prompt_equals_generation_prefix','all_prompt_labels_masked','contiguous_completion_labels',
        'target_labels_equal_input_ids','both_closing_tokens_supervised','earlier_closing_tokens_masked',
        'decoded_target_equals_reviewed_action','full_template_equals_trainer_ids')
    if len(audit['target_mask_checks'])!=31 or any(not all(c['label_audit'].get(k) for k in required) for c in audit['target_mask_checks']):
        raise ValueError('new label preflight incomplete')
    check=audit['joint_objective_self_check'];gradient=check['gradient_preflight']
    if (not all(check[k] for k in ('prompt_logits_masked','completion_logits_supervised','decision_logits_emphasized'))
        or check['model_parameters_updated'] or gradient['batch_count']!=31 or gradient['scale']!=256
        or gradient['optimizer_updates']!=0 or not gradient['all_gradients_finite'] or not gradient['gradients_cleared']):
        raise ValueError('joint gradient preflight failed')
    annotations=audit['weight_annotations']
    if len(annotations)!=31 or {a['action_index'] for a in annotations}!=set(range(31)):
        raise ValueError('weight coverage incomplete')
    for a in annotations:
        if any(w<=0 for w in a['target_weights']) or not math.isclose(a['normalizer'],1,abs_tol=1e-12):
            raise ValueError('completion weight failure')
        from training.agenttune.joint_choice_objective import joint_weights
        labels=[-100]*(max(a['label_positions'])+1);weights=[0.]*len(labels)
        for p,t,w in zip(a['label_positions'],a['target_token_ids'],a['base_target_weights'],strict=True):
            labels[p]=t;weights[p]=w
        original={**a,'target_weights':a['base_target_weights']}
        _,expected=joint_weights(labels,weights,original,config['decision_share'])
        if a['target_weights']!=expected['target_weights'] or a['decision_position']!=expected['decision_position']:
            raise ValueError('joint coefficients changed')
    from training.agenttune.sft_smoke import verify_edit_target_coverage
    verify_edit_target_coverage(annotations,mixed=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--artifact-repo',required=True)
    parser.add_argument('--config',type=Path);args=parser.parse_args()
    print(json.dumps(prepare(args.destination,args.artifact_repo,args.config)))
