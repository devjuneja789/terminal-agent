"""Pure validation of admitted continuation rows; never export prefix actions."""
from __future__ import annotations

from training.agenttune.prepare_sft import digest

IDS = {'C-B021-read-complete-v1', 'C-B026-choose-write-v1',
       'C-B041-preserve-file-v1', 'C-B046-explain-evidence-v1'}
ROWS_SHA256 = '03fc46a5cbfa41482ffbfda57ff8188bbfc761df998cc01a7f6e5bf0b803bb3b'


def verified_rows(refs: list[dict], rows: list[dict]) -> list[dict]:
    if len(refs) != 4 or {r['example_id'] for r in refs} != IDS or len(rows) != 15 or digest(rows) != ROWS_SHA256:
        raise ValueError('corrective source set or exported row hash changed')
    expected = []
    for ref in refs:
        if (ref['source_split'] != 'train-support' or not ref['training_eligible']
            or ref['human_review']['decision'] != 'approved' or ref['admission']['status'] != 'admitted'
            or ref['purpose'] == 'audit_only_failure_report'):
            raise ValueError('unadmitted corrective source')
        boundary = ref['context_message_count']
        if digest(ref['messages'][:boundary]) != ref['context_sha256']:
            raise ValueError('archived prefix changed')
        indices = [i for i, m in enumerate(ref['messages']) if i >= boundary and m['role'] == 'assistant']
        if indices != ref['proposed_target_message_indices']:
            raise ValueError('continuation boundary changed')
        for index in indices:
            expected.append((ref, index))
    if len(expected) != len(rows):
        raise ValueError('incorrect continuation count')
    for (ref, index), row in zip(expected, rows, strict=True):
        if (row['source_example_id'] != ref['example_id'] or row['source_task_id'] != ref['source_task_id']
            or row['source_message_index'] != index or row['context_message_count'] != ref['context_message_count']
            or row['prompt'] != ref['messages'][:index] or row['completion'] != [ref['messages'][index]]
            or row['source_record_sha256'] != ref['admission']['source_record_sha256']
            or not row['training_eligible'] or row['review_required']):
            raise ValueError('row differs from admitted continuation')
    return rows
