"""Recover complete paired trajectories from bounded HF console chunks."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def extract(log: Path, destination: Path, *, expected_count: int=8, expected_split: str='train-support') -> list[dict]:
    chunks = {}
    metadata = summary = None
    for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
        for marker in ('PAIRED_ENVIRONMENT=', 'PAIRED_TASK_CHUNK=', 'PAIRED_SUMMARY='):
            if marker not in line:
                continue
            value = json.loads(line.split(marker, 1)[1])
            if marker == 'PAIRED_ENVIRONMENT=':
                metadata = value
            elif marker == 'PAIRED_SUMMARY=':
                summary = value
            else:
                key, part, total = value['record_id'], value['part'], value['parts']
                if not 1 <= part <= total:
                    raise ValueError('invalid chunk numbering')
                entry = chunks.setdefault(key, {'parts': total, 'data': {}})
                if entry['parts'] != total or (part in entry['data'] and entry['data'][part] != value['data']):
                    raise ValueError('conflicting trajectory chunks')
                entry['data'][part] = value['data']
            break
    if metadata is None or summary is None:
        raise ValueError('missing paired environment or summary')
    expected = summary['record_ids']
    if len(expected) != expected_count or len(set(expected)) != expected_count or set(chunks) != set(expected) or metadata['record_ids'] != expected:
        raise ValueError(f'expected exactly {expected_count} paired records')
    records = []
    for key in expected:
        entry = chunks[key]
        if set(entry['data']) != set(range(1, entry['parts']+1)):
            raise ValueError(f'incomplete trajectory: {key}')
        encoded = ''.join(entry['data'][i] for i in range(1, entry['parts']+1))
        record = json.loads(base64.b64decode(encoded, validate=True))
        if record['record_id'] != key or key != f"{record['arm']}:{record['task_id']}" or record['split'] != expected_split:
            raise ValueError('paired record identity mismatch')
        records.append(record)
    destination.mkdir(parents=True, exist_ok=True)
    prefix = destination/summary['run_id']
    prefix.with_suffix('.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    prefix.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    prefix.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    title='# Paired held-out evaluation' if expected_split=='held-out-eval' else '# Paired training-support replay'
    lines = [title, '', summary['interpretation'], '',
             '| Arm | Contract passes | Invalid call rate | Average steps | Average latency (s) |',
             '|---|---:|---:|---:|---:|']
    for arm, metrics in summary['metrics'].items():
        lines.append(f"| {arm} | {metrics['successful_tasks']}/{metrics['task_count']} | {metrics['invalid_tool_call_rate']:.3f} | {metrics['average_tool_steps_per_task']:.2f} | {metrics['average_latency_seconds']:.2f} |")
    lines += ['', '| Arm | Task | Contract | Failure |', '|---|---|---|---|']
    for r in records:
        lines.append(f"| {r['arm']} | {r['task_id']} | {'PASS' if r['success'] else 'FAIL'} | {r['failure_category']} |")
    lines += ['', 'Contract passes require trace review, especially factual explanations.',
        'Frozen held-out tasks; no training use permitted.' if expected_split=='held-out-eval' else 'These are training tasks; this is not held-out performance.', '']
    prefix.with_suffix('.md').write_text('\n'.join(lines))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-count',type=int,default=8)
    parser.add_argument('--split',choices=['train-support','held-out-eval'],default='train-support')
    args = parser.parse_args()
    print(f'Extracted {len(extract(args.log, args.output_dir,expected_count=args.expected_count,expected_split=args.split))} complete records')
