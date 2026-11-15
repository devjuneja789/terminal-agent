"""Prepare explicit HF payloads and a checksum manifest; never publish them."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination: Path) -> dict:
    freeze_path=ROOT/'configs/submission_candidate.json'
    frozen=json.loads(freeze_path.read_text())
    final_index=json.loads((ROOT/'submission/final_artifacts.json').read_text())
    if final_index['candidate_freeze_sha256']!=hashlib.sha256(freeze_path.read_bytes()).hexdigest():
        raise ValueError('release results differ from frozen candidate')
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('release directory must be empty')
    destination.mkdir(parents=True,exist_ok=True)
    support=destination/'support-data';logs=destination/'evaluation-logs'
    support.mkdir();logs.mkdir()
    # Exact admitted sources and review/provenance; no rejected examples become targets.
    relatives=['data/sft/tiny_aya_corrective_v1.jsonl',
        'data/corrective/support_corrective.v1.admitted.jsonl',
        'data/corrective/support_corrective.v1.review.jsonl',
        'data/corrective/support_corrective.v1.owner-review-receipt.json']
    for relative in relatives:
        source=ROOT/relative;target=support/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    # Preserve produced data/provenance without admitting archive rows as targets.
    for source in (ROOT/'data').rglob('*'):
        if not source.is_file() or source.suffix not in ('.json','.jsonl','.md'):continue
        target=support/'archive'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    archives=list((ROOT/'research/results').glob('tiny-aya-*/*'))+list((ROOT/'research/results').glob('auditkit-*/*'))
    for source in archives:
        if not source.is_file() or source.suffix not in ('.json','.jsonl','.md'):continue
        target=logs/'archive'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    # Host the benchmark and its standalone snapshot beside evaluation evidence.
    for source in (ROOT/'evals/benchmark').rglob('*'):
        relative=source.relative_to(ROOT/'evals/benchmark')
        if ('results' in relative.parts or '__pycache__' in relative.parts
            or not source.is_file() or source.suffix not in ('.py','.json','.jsonl','.md','.txt','.toml')):continue
        target=logs/'benchmark'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for relative in final_index['release_evidence_paths']:
        source=ROOT/relative
        if source.suffix=='.log' or not source.is_file():raise ValueError('invalid evidence path: '+relative)
        target=logs/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copy2(ROOT/'configs/submission_candidate.json',logs/'candidate.json')
    (support/'README.md').write_text('''---
language: en
task_categories:
- text-generation
tags:
- terminal-slm
- support-only
---
# Terminal-SLM reviewed support targets

Four owner-reviewed, separately admitted corrective demonstrations, exported as
15 continuation-only action targets. These are manually authored/reviewed
references, not successful Tiny Aya-generated trajectories. Failed prefixes are
context only. Two truthful-failure reports remain permanently excluded.

`archive/` retains earlier experimental datasets, rejected examples and review
history for provenance. It is not an admitted training corpus: never merge
archive rows into the current 15-target export. Pending/rejected historical
metadata remains unchanged rather than being silently relabelled.

This release contains no held-out training targets. See the review receipt and
admission record for provenance. Exact approved-target hash:
`03fc46a5cbfa41482ffbfda57ff8188bbfc761df998cc01a7f6e5bf0b803bb3b`.
Code examples describe synthetic benchmark repositories. See the source project
for full admission, masking and reproduction instructions. Licensing must be
confirmed by the project owner before publication.
''')
    (logs/'README.md').write_text('''---
language: en
tags:
- terminal-slm
- evaluation-only
- not-for-training
---
# Terminal-SLM evaluation evidence

Actual model predictions, executable validation outputs, versioned configs and
AuditKIT recorded-episode reports. Held-out episodes are **evaluation-only and
prohibited from use as this project's training data**. Raw model text, failures
and harness final-response attribution are preserved. No synthetic successful
benchmark runs are included. Metrics describe a small deterministic benchmark,
not unrestricted repository repair capability. See the included final report
and source repository for the frozen protocol and limitations.
`archive/` retains the prior Tiny Aya diagnostic results, failures and review
history; these are not additional final held-out runs or model-selection data.
`benchmark/` contains the fixed task definitions and executable snapshot.
''')
    model_cards=destination/'model-cards';model_cards.mkdir()
    results=[]
    for source in sorted((ROOT/'research/results').glob('tiny-aya-*/*.result.json')):
        result=json.loads(source.read_text())
        if not result.get('config',{}).get('artifact_repo'):continue
        repo=result['config']['artifact_repo'];slug=repo.split('/')[-1]
        artifact_commit=result.get('artifact_commit')
        if artifact_commit is None:
            replay=json.loads((ROOT/'research/results/tiny-aya-paired-replay/20261113T223053Z.metadata.json').read_text())['config']['replay']
            if replay['adapter_repo']!=repo or replay['source_training_run_id']!=result['run_id']:
                raise ValueError('adapter artifact provenance missing: '+repo)
            artifact_commit=replay['adapter_revision']
        selected=repo==frozen['adapter_repo'] and artifact_commit==frozen['adapter_revision']
        card='''---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
tags:
- terminal-slm
- agenttune
- experimental
---
# Terminal-SLM experimental adapter

This repository contains a LoRA adapter, not the base model weights. It derives
from Cohere Labs Tiny Aya Global at
`af89d219b53ed9b13b8a4645f9c8028973510324`. Retain Cohere attribution,
CC-BY-NC-4.0 noncommercial terms and its acceptable-use requirements.
Base: https://huggingface.co/CohereLabs/tiny-aya-global

Source/reproduction: https://github.com/devjuneja789/terminal-agent

'''
        card+=f"Run: `{result['run_id']}`. Training config: `{result['config']['config_id']}`.\n"
        card+=f"Pinned artifact commit before this metadata update: `{artifact_commit}`.\n"
        card+=f"Adapter subfolder: `{result['run_id']}/adapter`.\n\n"
        card+=('Selected submission candidate; final paired metrics are in the source submission report.\n' if selected
            else 'Diagnostic experiment; not the selected submission candidate. Do not treat training loss or a smoke pass as task-success improvement.\n')
        card+='\nExperimental terminal-use research only. No claim of reliable autonomous coding or secure shell sandboxing.\n'
        (model_cards/(slug+'.md')).write_text(card)
        results.append({'repo_id':repo,'artifact_commit':artifact_commit,
            'adapter_subfolder':result['run_id']+'/adapter','selected':selected,
            'model_card':str(Path('model-cards')/(slug+'.md'))})
    paths=[p for p in destination.rglob('*') if p.is_file()]
    manifest={'status':'prepared_for_owner_review_not_published','payload_directory':str(destination),'candidate':frozen,
        'model_repositories':results,'proposed_support_repo':'devjuneja789/terminal-slm-reviewed-support',
        'proposed_evaluation_repo':'devjuneja789/terminal-slm-evaluation-evidence',
        'required_public_collections':['Terminal-SLM models and adapters','Terminal-SLM data and evaluation'],
        'source_repository':'https://github.com/devjuneja789/terminal-agent',
        'files':{str(p.relative_to(destination)):{'bytes':p.stat().st_size,
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}}
    (destination/'release-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();result=prepare(args.destination)
    print(json.dumps({'status':result['status'],'files':len(result['files']),
        'model_repositories':len(result['model_repositories'])}))
