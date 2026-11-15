"""Stage approved references and SFT code without benchmark or held-out files."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from training.agenttune.prepare_sft import ROOT, action_rows, digest, read_rows, verified_references


def prepare(destination: Path, artifact_repo: str, config_path: Path | None = None) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', artifact_repo):
        raise ValueError('artifact repo must be namespace/name')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('bundle directory must be empty')
    data = ROOT/'data/sft'
    raw = read_rows(data/'gold_support.jsonl')
    reviews = read_rows(data/'gold_support.review.jsonl')
    refs = verified_references(raw, reviews, read_rows(data/'gold_support.approved.jsonl'))
    ids = {row['example_id'] for row in refs}
    for relative in ['training/__init__.py', 'training/agenttune/__init__.py',
                     'training/agenttune/prepare_sft.py', 'training/agenttune/sft_smoke.py',
                     'training/agenttune/weighted_loss.py',
                     'training/agenttune/run_sft_job.sh']:
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
    shutil.copytree(ROOT/'tools', destination/'tools', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (destination/'data').mkdir()
    for name, values in [('source_records', [r for r in raw if r['example_id'] in ids]),
                         ('reviews', [r for r in reviews if r['example_id'] in ids]), ('approved', refs)]:
        (destination/f'data/{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in values))
    config = json.loads((config_path or ROOT/'configs/tiny_aya_sft_smoke.json').read_text())
    config['artifact_repo'] = artifact_repo
    (destination/'configs').mkdir()
    (destination/'configs/tiny_aya_sft_smoke.json').write_text(json.dumps(config, indent=2)+'\n')
    manifest = {'source_example_ids': sorted(ids), 'action_target_count': len(action_rows(refs)),
                'rows_sha256': digest(action_rows(refs)), 'config_sha256': digest(config),
                'heldout_accessed': False, 'rejected_references_included': False}
    (destination/'data/manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--artifact-repo', required=True)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination, args.artifact_repo, args.config)))
