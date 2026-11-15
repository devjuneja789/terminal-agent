"""Independently audit the eight native-stop replay records; never run inference."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.agenttune.evaluate_heldout import _aggregate

LEGACY = ROOT / 'research/results/tiny-aya-weighted-replay/20261114T161125Z'
STOPS = [3, 6, 261001]
ADDED = ('eos_token_ids', 'capture_stop_audit', 'stop_policy_version', 'source_stop_preflight_run_id')


def read(prefix: Path, suffix: str):
    path = prefix.with_suffix(suffix)
    if suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines()]
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def analyze(prefix: Path) -> dict:
    records, metadata, summary = (read(prefix, s) for s in ('.jsonl', '.metadata.json', '.summary.json'))
    old_records, old_metadata = (read(LEGACY, s) for s in ('.jsonl', '.metadata.json'))
    config = metadata['config']
    restored = copy.deepcopy(config)
    restored['config_id'] = old_metadata['config']['config_id']
    for field in ADDED:
        restored['replay'].pop(field)
    require(restored == old_metadata['config'], 'configuration changed beyond versioned stopping/telemetry')
    require(metadata['config_sha256'] == hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(), 'config digest mismatch')
    require(config['config_id'] == 'terminal-slm-native-stops-support-replay-v1', 'wrong runtime version')
    require(metadata['generation_stop_ids'] == STOPS and metadata['upstream_generation_eos_token_ids'] == 3
            and metadata['generation_stop_audit_enabled'], 'incorrect native stopping configuration')
    require(metadata['quantization'] == old_metadata['quantization'], 'quantization changed')
    require(metadata['tokenizer_equivalence'] == old_metadata['tokenizer_equivalence'], 'tokenizer changed')
    require(metadata['heldout_accessed'] is False and summary['heldout_accessed'] is False, 'held-out contamination')
    expected = [f'{arm}:{task}' for arm in ('base', 'adapter') for task in ('B021', 'B026', 'B041', 'B046')]
    require([r['record_id'] for r in records] == expected == metadata['record_ids'] == summary['record_ids'], 'incomplete paired replay')
    previous = {r['record_id']: r for r in old_records}
    audits = []
    for record in records:
        key = record['record_id']
        old = previous[key]
        trace = record['agenttune_trace']
        for field in ('split', 'task_id', 'model_id', 'model_revision', 'snapshot_sha256', 'success_condition'):
            require(record[field] == old[field], f'{key}: {field} drifted')
        require(record['split'] == 'train-support' and record['training_target_eligible'] is False, 'unsafe training admission')
        require(record['validation']['command'] == old['validation']['command'], f'{key}: validator changed')
        require(trace['model_prompts'][0] == old['agenttune_trace']['model_prompts'][0], f'{key}: initial prompt changed')
        generations = record['generation_audit']
        require(len(generations) == len(trace['model_outputs']) > 0, f'{key}: missing generation telemetry')
        require(sum(g['generated_tokens'] for g in generations) == record['token_usage']['generated_tokens'], f'{key}: token accounting mismatch')
        endings = Counter()
        for generation in generations:
            ids = generation['generated_token_ids']
            cap = config['generation']['max_new_tokens']
            closed = bool(ids and ids[-1] in STOPS)
            require(generation['configured_stop_ids'] == STOPS, f'{key}: stop union not applied')
            require(len(ids) == generation['generated_tokens'] and ids and generation['last_token_id'] == ids[-1], f'{key}: token ID mismatch')
            require(generation['stopped_on_configured_id'] == closed and generation['hit_token_cap'] == (len(ids) == cap), f'{key}: incorrect ending flags')
            require(0 < len(ids) <= cap and (closed or len(ids) == cap), f'{key}: unaccounted generation termination')
            counts = [ids.count(i) for i in STOPS]
            require(counts == [generation['upstream_eos_count'], generation['turn_end_count'], generation['response_end_count']], f'{key}: closing count mismatch')
            require(sum(counts) == int(closed), f'{key}: tokens emitted after a recognized closing')
            endings[str(ids[-1]) if closed else 'length_cap'] += 1
        writes = [e for e in trace['tool_executions'] if e.get('tool') == 'write_file']
        audits.append({'record_id': key, 'success': record['success'], 'failure_category': record['failure_category'],
            'selected_actions_equal_legacy': trace['selected_actions'] == old['agenttune_trace']['selected_actions'],
            'decoded_outputs_equal_legacy': trace['model_outputs'] == old['agenttune_trace']['model_outputs'],
            'generation_count': len(generations), 'ending_counts': dict(endings),
            'generated_tokens': record['token_usage']['generated_tokens'], 'write_file_executions': len(writes),
            'changed_files': record['support_audit']['changed_files'],
            'validation_exit_code': record['validation']['exit_code'], 'criterion_failures': record['criterion_failures'],
            'final_response': record['final_response'], 'final_response_source': record['final_response_source']})
    metrics = {arm: _aggregate([r for r in records if r['arm'] == arm]) for arm in ('base', 'adapter')}
    require(metrics == summary['metrics'], 'summary does not match raw records')
    comparison = {}
    for arm in ('base', 'adapter'):
        new = [r for r in records if r['arm'] == arm]
        old = [r for r in old_records if r['arm'] == arm]
        selected = [a for a in audits if a['record_id'].startswith(arm + ':')]
        comparison[arm] = {'legacy_metrics': _aggregate(old), 'native_metrics': metrics[arm],
            'legacy_generated_tokens': sum(r['token_usage']['generated_tokens'] for r in old),
            'native_generated_tokens': sum(r['token_usage']['generated_tokens'] for r in new),
            'native_ending_counts': dict(sum((Counter(a['ending_counts']) for a in selected), Counter())),
            'native_write_file_executions': sum(a['write_file_executions'] for a in selected)}
    return {'run_id': summary['run_id'], 'config_id': config['config_id'], 'config_sha256': metadata['config_sha256'],
        'runtime_stop_gate_passed': True, 'stop_ids': STOPS, 'config_equivalence_except_versioned_stops_and_telemetry': True,
        'heldout_accessed': False, 'training_performed': False, 'training_admission_performed': False,
        'comparison': comparison, 'task_audits': audits,
        'interpretation': 'Runtime validation on four previously trained support tasks, not generalization. Stopping efficiency is not a training gain.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--recovered-log', type=Path)
    args = parser.parse_args()
    result = analyze(args.prefix)
    path = args.prefix.with_suffix('.analysis.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    gate = {k: result[k] for k in ('run_id', 'config_id', 'config_sha256', 'runtime_stop_gate_passed',
        'stop_ids', 'heldout_accessed', 'training_performed', 'training_admission_performed')}
    gate['analysis_path'] = str(path.relative_to(ROOT) if path.is_absolute() else path)
    gate['analysis_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.log:
        job = re.findall(r'id=([0-9a-f]{24}) name=terminal-slm-native-stops-support-replay-v1', args.log.read_text())
        require(len(set(job)) == 1, 'cannot identify exactly one native replay GPU job')
        gate['hf_job_id'] = job[0]
        gate['log_sha256'] = hashlib.sha256(args.log.read_bytes()).hexdigest()
    if args.recovered_log:
        gate['recovered_log_sha256'] = hashlib.sha256(args.recovered_log.read_bytes()).hexdigest()
    gate['artifacts_sha256'] = {suffix: hashlib.sha256(args.prefix.with_suffix(suffix).read_bytes()).hexdigest()
        for suffix in ('.jsonl', '.metadata.json', '.summary.json')}
    gate['source_sha256'] = {str(source): hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
        for source in (Path('training/agenttune/paired_replay.py'), Path('training/agenttune/prepare_native_stop_replay.py'),
            Path('training/agenttune/hf_gpu_smoke.py'), Path('training/agenttune/evaluate_heldout.py'),
            Path('scripts/analyze_native_stop_replay.py'))}
    args.prefix.with_suffix('.runtime-gate.json').write_text(json.dumps(gate, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('run_id', 'runtime_stop_gate_passed', 'stop_ids', 'training_performed', 'heldout_accessed')}))
    print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
