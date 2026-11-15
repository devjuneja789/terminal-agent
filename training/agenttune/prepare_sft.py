"""Verify owner approval and export one completion-only example per action."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUPPORT_IDS = frozenset({'B021', 'B026', 'B041', 'B046'})


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def verified_references(raw: list[dict], reviews: list[dict], approved: list[dict]) -> list[dict]:
    sources = {row['example_id']: row for row in raw}
    dispositions = {row['example_id']: row for row in reviews}
    if len(sources) != len(raw) or len(dispositions) != len(reviews):
        raise ValueError('duplicate source or review IDs')
    seen = set()
    for row in approved:
        key = row['example_id']
        if key in seen:
            raise ValueError('duplicate approved ID')
        seen.add(key)
        source, review = sources[key], dispositions[key]
        if source['source_split'] != 'train-support' or source['source_task_id'] not in SUPPORT_IDS:
            raise ValueError('only the four approved support tasks are allowed')
        if review['decision'] != 'approved' or review['source_record_sha256'] != digest(source):
            raise ValueError('missing approval or changed source record')
        expected = copy.deepcopy(source)
        expected['reference_audit'].update(human_review_status='approved', training_eligible=True)
        expected['human_review'] = review
        if row != expected:
            raise ValueError('approved reference differs from reviewed source')
    if not approved:
        raise ValueError('no approved references')
    return approved


def action_rows(references: list[dict]) -> list[dict]:
    rows = []
    for ref in references:
        messages = ref['messages']
        for index, message in enumerate(messages):
            if message['role'] != 'assistant':
                continue
            action = json.loads(message['content'])
            if not isinstance(action, list) or len(action) != 1:
                raise ValueError('each target must contain one action')
            from tools.definitions import validate_tool_arguments
            selected = action[0]
            if set(selected) != {'tool_name', 'parameters'}:
                raise ValueError('invalid action fields')
            if selected['tool_name'] == 'finish':
                if set(selected['parameters']) != {'answer'} or not isinstance(selected['parameters']['answer'], str):
                    raise ValueError('invalid finish')
            else:
                validate_tool_arguments(selected['tool_name'], selected['parameters'])
            rows.append({'prompt': copy.deepcopy(messages[:index]), 'completion': [copy.deepcopy(message)],
                         'source_example_id': ref['example_id'], 'source_task_id': ref['source_task_id']})
    return rows


def main() -> int:
    root = ROOT / 'data/sft'
    refs = verified_references(read_rows(root/'gold_support.jsonl'), read_rows(root/'gold_support.review.jsonl'), read_rows(root/'gold_support.approved.jsonl'))
    rows = action_rows(refs)
    output = root/'tiny_aya_sft_smoke.jsonl'
    output.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')
    manifest = {'kind': 'approved_support_sft_smoke', 'source_examples': [r['example_id'] for r in refs],
                'trajectory_count': len(refs), 'action_target_count': len(rows), 'rows_sha256': digest(rows),
                'loss': 'completion_only', 'heldout_accessed': False,
                'purpose': 'pipeline smoke only; four references cannot establish generalization'}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
