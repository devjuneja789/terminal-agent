"""Build explicit HF viewer configs and display copies without changing raw evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCES = {
    'benchmark-heldout': 'benchmark/heldout_eval.jsonl',
    'benchmark-support': 'benchmark/train_support.jsonl',
    'final-heldout-runs': 'evidence/research/results/tiny-aya-final-heldout/20261115T071710Z.jsonl',
    'fit300-contexts': 'evidence/research/results/tiny-aya-fit300-contexts/20261115T071048Z.context.jsonl',
    'fit300-predictions': 'evidence/research/results/tiny-aya-fit300-contexts/20261115T071048Z.prediction.jsonl',
    'joint60-contexts': 'evidence/research/results/tiny-aya-joint-contexts/20261115T063016Z.context.jsonl',
    'joint60-predictions': 'evidence/research/results/tiny-aya-joint-contexts/20261115T063016Z.prediction.jsonl',
    'task-start-support-runs': 'evidence/research/results/tiny-aya-task-start-replay/20261115T124045Z.jsonl',
    'task-start-training-history': 'evidence/research/results/tiny-aya-task-start-sft/20261115T115312Z.training-history.jsonl',
}


def prepare(root: Path) -> dict:
    # Check the complete source list before making any changes.
    source_rows = {}
    for name, relative in SOURCES.items():
        path = root / relative
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f'Expected nonempty object rows: {relative}')
        source_rows[name] = rows
    viewer = root / 'viewer'
    viewer.mkdir(exist_ok=True)
    manifest = {'kind': 'viewer_display_copies_v1', 'raw_evidence_modified': False, 'configs': {}}
    configs = []
    for name, relative in SOURCES.items():
        rows = source_rows[name]
        # Every display cell is a string. Encode values as JSON so type and
        # structure can be recovered, even when one path alternates dict/str.
        # Rows remain ordered; absent fields stay absent rather than invented.
        display = [{key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                    for key, value in row.items()} for row in rows]
        recovered = [{key: json.loads(value) for key, value in row.items()} for row in display]
        if recovered != rows:
            raise ValueError(f'Non-lossless display conversion: {relative}')
        target = viewer / f'{name}.jsonl'
        target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in display))
        columns = sorted(set().union(*(row.keys() for row in display)))
        configs += [f'- config_name: {name}']
        if name == 'final-heldout-runs':
            configs += ['  default: true']
        configs += ['  data_files:', '  - split: evidence', f'    path: viewer/{name}.jsonl', '  features:']
        for column in columns:
            configs += [f'  - name: {column}', '    dtype: string']
        manifest['configs'][name] = {
            'source': relative, 'viewer': f'viewer/{name}.jsonl', 'row_count': len(rows),
            'source_sha256': hashlib.sha256((root / relative).read_bytes()).hexdigest(),
            'viewer_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    header = ['---', 'language: en', 'license: cc-by-nc-4.0', 'tags:', '- terminal-slm',
              '- evaluation-only', '- not-for-training', 'configs:', *configs, '---', '']
    body = '''# Terminal-SLM evaluation evidence

This dataset contains the EXP-025 matched held-out evaluation and the later
EXP-027 support-task experiment, with benchmark definitions, recorded outputs,
validation evidence, and AuditKIT reports.

## Using the viewer

Select a configuration from the viewer dropdown. Each configuration contains
one source file, so benchmark tasks, model predictions, agent trajectories, and
training-history rows are never combined into one schema. The default is
`final-heldout-runs` (20 recorded base/adapter episodes).

The viewer reads display copies under `viewer/`. Each cell is JSON-encoded text
with an explicit string feature type; this handles nested tool arguments and
observations whose structures vary between actions. For example, `success` is
shown as `true` or `false`, and nested trajectory data is shown as JSON text.
Apply `json.loads(cell)` to recover a value's original type. Missing source
fields remain missing. These are display copies, not new runs or training data.

The original JSONL files under `benchmark/` and `evidence/` remain byte-for-byte
unchanged and are the authoritative records. Summary JSON, model metadata,
reports and other artifacts remain downloadable through Files and versions;
they are not automatically pooled into the viewer. `viewer-manifest.json`
records the source paths, row counts and hashes for every display copy.

## Interpretation and data boundaries

EXP-025: base 1/10 versus adapter 2/10 contract passes; both 0/5 repairs. The
adapter's extra explanation pass was factually unreliable. EXP-027: base 0/4
versus adapter 2/4 on known support tasks; that adapter has no held-out result.
These experiments do not establish reliable autonomous coding.

Held-out records are evaluation-only and must never become this project's
training data. Support replay records here are evidence, not admitted training
targets. Reviewed training data is hosted separately in
[terminal-slm-reviewed-support](https://huggingface.co/datasets/devjuneja789/terminal-slm-reviewed-support).

License: CC-BY-NC-4.0. Tiny Aya-derived material remains subject to Cohere Labs'
base-model terms and acceptable-use policy. Source and reproduction:
https://github.com/devjuneja789/terminal-agent
'''
    (root / 'README.md').write_text('\n'.join(header) + body)
    (root / 'viewer-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('submission/hf-upload/evaluation-logs'))
    args = parser.parse_args()
    result = prepare(args.root)
    print(json.dumps({'configs': len(result['configs']),
                      'rows': {key: value['row_count'] for key, value in result['configs'].items()},
                      'raw_evidence_modified': result['raw_evidence_modified']}, indent=2))
