"""Stage four admitted corrective sources, 15 fixed targets and passed gates."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from training.agenttune.prepare_corrective_preflight import verify_export
from training.agenttune.prepare_sft import ROOT, digest, read_rows
from training.agenttune.corrective_training_data import verified_rows


def prepare(destination: Path, artifact_repo: str, config_path: Path | None = None) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', artifact_repo):
        raise ValueError('invalid private artifact repository')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('bundle directory must be empty')
    rows, admission = verify_export()
    refs = read_rows(ROOT/'data/corrective/support_corrective.v1.admitted.jsonl')
    verified_rows(refs, rows)
    labels = json.loads((ROOT/'data/sft/tiny_aya_corrective_v1.preflight.json').read_text())
    gate = json.loads((ROOT/'research/results/tiny-aya-native-stop-replay/20261114T220329Z.runtime-gate.json').read_text())
    analysis = ROOT/gate['analysis_path']
    import hashlib
    if (not labels['all_label_checks_passed'] or not labels['all_weight_annotations_passed']
        or labels['rows_sha256'] != digest(rows) or not gate['runtime_stop_gate_passed']
        or gate['stop_ids'] != [3, 6, 261001] or gate['heldout_accessed']
        or hashlib.sha256(analysis.read_bytes()).hexdigest() != gate['analysis_sha256']):
        raise ValueError('label or native-stop gate failed/changed')
    config = json.loads((ROOT/'configs/tiny_aya_corrective_sft.json').read_text())
    if config_path and json.loads(config_path.read_text()) != config:
        raise ValueError('corrective training configuration is fixed')
    config.update(artifact_repo=artifact_repo)
    for relative in ('training/__init__.py', 'training/agenttune/__init__.py',
        'training/agenttune/prepare_sft.py', 'training/agenttune/sft_smoke.py',
        'training/agenttune/weighted_loss.py', 'training/agenttune/corrective_preflight.py',
        'training/agenttune/corrective_training_data.py', 'training/agenttune/run_sft_job.sh'):
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
    shutil.copytree(ROOT/'tools', destination/'tools', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (destination/'data').mkdir(); (destination/'configs').mkdir()
    for name, values in (('approved', refs), ('corrective_rows', rows)):
        (destination/f'data/{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values))
    manifest = {'rows_sha256': digest(rows), 'refs_sha256': digest(refs),
        'source_example_ids': [r['example_id'] for r in refs], 'action_target_count': 15,
        'config_sha256': digest(config), 'admission_rows_sha256': admission['rows_sha256'],
        'label_gate': labels, 'runtime_gate': gate, 'old_gold_merged': False, 'heldout_accessed': False}
    (destination/'data/manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (destination/'configs/tiny_aya_sft_smoke.json').write_text(json.dumps(config, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--artifact-repo', required=True)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination, args.artifact_repo, args.config)))
