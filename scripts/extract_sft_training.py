"""Recover full SFT metrics and history from lossless console chunks."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


def extract(log: Path, destination: Path) -> Path:
    parts = {}
    total = None
    for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
        marker = 'SFT_TRAINING_RESULT_CHUNK='
        if marker not in line:
            continue
        row = json.loads(line.split(marker, 1)[1])
        if total is not None and total != row['parts']:
            raise ValueError('inconsistent training chunk count')
        total = row['parts']
        index = row['part']
        if not 1 <= index <= total or (index in parts and parts[index] != row['data']):
            raise ValueError('conflicting training chunks')
        parts[index] = row['data']
    if total is None or set(parts) != set(range(1, total+1)):
        raise ValueError('missing or incomplete training result chunks')
    result = json.loads(base64.b64decode(''.join(parts[i] for i in range(1, total+1)), validate=True))
    destination.mkdir(parents=True, exist_ok=True)
    prefix = destination/result['run_id']
    path = prefix.with_suffix('.result.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    prefix.with_suffix('.training-history.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in result['training_audit']['log_history']))
    prefix.with_suffix('.provenance.json').write_text(json.dumps({
        'source_log': str(log), 'source_log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
        'complete_training_chunks': True, 'artifact_commit': result['artifact_commit']}, indent=2)+'\n')
    return path.resolve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--result-pointer', type=Path)
    args = parser.parse_args()
    result = extract(args.log, args.output_dir)
    if args.result_pointer:
        args.result_pointer.write_text(str(result)+'\n')
    print(f'Full training result: {result}')
