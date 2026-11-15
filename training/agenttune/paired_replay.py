"""GPU-only paired base/adapter replay; no held-out access or training."""
from __future__ import annotations

import base64
import gc
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.benchmark.loader import BENCHMARK_ROOT, load_benchmark
from tools.definitions import get_tool_schemas
from training.agenttune.evaluate_heldout import _aggregate, _hardware, _run_one
from training.agenttune.hf_gpu_smoke import EVAL_CONFIG, MODEL_ID, MODEL_REVISION, _canonical_sha256, verify_frozen_config, verify_runtime_versions
from training.agenttune.support_rewards import score_support_rollout


def emit_record(record: dict) -> None:
    encoded = base64.b64encode(json.dumps(record, ensure_ascii=False, separators=(',', ':')).encode()).decode()
    chunks = [encoded[i:i+12000] for i in range(0, len(encoded), 12000)]
    for index, chunk in enumerate(chunks, 1):
        print('PAIRED_TASK_CHUNK='+json.dumps({'record_id': record['record_id'],
            'part': index, 'parts': len(chunks), 'data': chunk}, separators=(',', ':')), flush=True)


def verify_tokenizers(base, saved) -> dict:
    # save_pretrained moves the sole named default template into a .jinja file.
    # Compare the selected template text, not dict-vs-string storage containers.
    template = base.get_chat_template()
    if template != saved.get_chat_template() or base.get_vocab() != saved.get_vocab():
        raise RuntimeError('adapter tokenizer differs from base: effective template or vocabulary mismatch')
    special_ids = {name: getattr(base, name) for name in ('bos_token_id', 'eos_token_id', 'pad_token_id')}
    if any(getattr(saved, name) != value for name, value in special_ids.items()):
        raise RuntimeError('adapter tokenizer special token IDs differ from base')
    return {'effective_template_sha256': _canonical_sha256(template),
            'special_token_ids': special_ids, 'vocabulary_equal': True}


def generation_stops(replay: dict,tokenizer,upstream_eos):
    """Keep legacy behavior unless an explicit, complete native union is selected."""
    if 'eos_token_ids' not in replay:
        return tokenizer.eos_token_id
    ids=replay['eos_token_ids']
    if (not isinstance(ids,list) or not ids
        or any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in ids) or len(set(ids))!=len(ids)):
        raise ValueError('invalid explicit stop IDs')
    native=upstream_eos if isinstance(upstream_eos,list) else [upstream_eos]
    response=tokenizer.convert_tokens_to_ids('<|END_RESPONSE|>')
    if response is None or response==getattr(tokenizer,'unk_token_id',None):
        raise ValueError('native response closing token unavailable')
    expected=set(native+[tokenizer.eos_token_id,response])
    if None in expected or set(ids)!=expected:
        raise ValueError('explicit stop policy must preserve all native closing IDs')
    return list(ids)


def capture_generations(model,max_new_tokens: int) -> list[dict]:
    """Transparent opt-in logging; forward every generate argument unchanged."""
    original=model.generate
    audit=[]
    def generate(*args,**kwargs):
        prompt=kwargs['input_ids'] if 'input_ids' in kwargs else args[0]
        length=prompt.shape[-1]
        output=original(*args,**kwargs)
        ids=output[0,length:].tolist()
        stops=model.generation_config.eos_token_id
        stops=list(stops) if isinstance(stops,(list,tuple)) else [stops]
        audit.append({'generated_token_ids':ids,'generated_tokens':len(ids),'prompt_tokens':length,
            'configured_stop_ids':stops,'last_token_id':ids[-1] if ids else None,
            'stopped_on_configured_id':bool(ids and ids[-1] in stops),
            'response_end_count':ids.count(261001),'turn_end_count':ids.count(6),'upstream_eos_count':ids.count(3),
            'hit_token_cap':len(ids)==max_new_tokens})
        return output
    model.generate=generate
    return audit


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('paired replay requires an HF CUDA Job; CPU inference is disabled')
    schemas = get_tool_schemas()
    verify_frozen_config(schemas)
    verify_runtime_versions(torch)
    replay = EVAL_CONFIG['replay']
    for name, version in replay['extra_runtime'].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'runtime mismatch for {name}')
    manifest = json.loads((BENCHMARK_ROOT/'manifest.json').read_text())
    if set(manifest['splits']) != {'train-support'} or (BENCHMARK_ROOT/'heldout_eval.jsonl').exists():
        raise RuntimeError('bundle must contain train-support only')
    tasks = list(load_benchmark('train-support'))
    if [t.task_id for t in tasks] != replay['task_ids'] or len(tasks) != 4:
        raise RuntimeError('unexpected paired task selection')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    saved_tokenizer = AutoTokenizer.from_pretrained(replay['adapter_repo'],
        revision=replay['adapter_revision'], subfolder=replay['adapter_subfolder'])
    tokenizer_check = verify_tokenizers(tokenizer, saved_tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    upstream_eos=GenerationConfig.from_pretrained(MODEL_ID,revision=MODEL_REVISION).eos_token_id
    stop_ids=generation_stops(replay,tokenizer,upstream_eos)
    quant_args = dict(replay['quantization'])
    quant_args['bnb_4bit_compute_dtype'] = torch.float16
    quant = BitsAndBytesConfig(**quant_args)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    metadata = {'run_kind': 'paired_training_support_replay', 'run_id': run_id,
        'task_ids': replay['task_ids'], 'arms': replay['arms'], 'config': EVAL_CONFIG,
        'config_sha256': _canonical_sha256(EVAL_CONFIG), 'hardware': _hardware(torch),
        'tokenizer_equivalence': tokenizer_check,
        'generation_stop_ids':stop_ids,'upstream_generation_eos_token_ids':upstream_eos,
        'generation_stop_audit_enabled':replay.get('capture_stop_audit',False),
        'quantization': quant.to_dict(), 'heldout_accessed': False,
        'interpretation': 'training-example replay; contract metrics require factual trace review; not generalization',
        'record_ids': [f'{arm}:{t.task_id}' for arm in replay['arms'] for t in tasks]}
    print('PAIRED_ENVIRONMENT='+json.dumps(metadata, separators=(',', ':')), flush=True)
    records = []
    for arm in replay['arms']:
        set_seed(replay['seed'])
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
            dtype=torch.float16, device_map={'': 0}, quantization_config=quant,
            attn_implementation='eager')
        if arm == 'adapter':
            model = PeftModel.from_pretrained(model, replay['adapter_repo'],
                revision=replay['adapter_revision'], subfolder=replay['adapter_subfolder'])
        # Apply the same tokenizer stop/pad settings to both inference arms.
        model.generation_config.eos_token_id = stop_ids
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        generation_audit=capture_generations(model,EVAL_CONFIG['generation']['max_new_tokens']) if replay.get('capture_stop_audit',False) else []
        for task in tasks:
            set_seed(replay['seed'])
            audit_start=len(generation_audit)
            record, _ = _run_one(task, tokenizer, model, torch, schemas,
                run_id=run_id, metadata=metadata, capture_support_audit=True)
            record.update(run_kind='paired_training_support_replay', arm=arm,
                record_id=f'{arm}:{task.task_id}', training_target_eligible=False)
            record['support_reward'] = score_support_rollout(task, record)
            if replay.get('capture_stop_audit',False):
                record['generation_audit']=generation_audit[audit_start:]
            # Completed nonzero shell exits are observations, not executor faults.
            if record['failure_category'] == 'tool_execution':
                actual_fault = any(not e.get('schema_valid', True) or
                    not isinstance(e.get('result'), dict) or e['result'].get('error') is not None
                    for e in record['agenttune_trace']['tool_executions'])
                if not actual_fault:
                    record['original_failure_category'] = 'tool_execution'
                    record['failure_category'] = 'task_contract_failure'
            records.append(record)
            emit_record(record)
            print(f"{arm} {task.task_id}: contract={'PASS' if record['success'] else 'FAIL'} ({record['failure_category']})", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    metrics = {arm: _aggregate([r for r in records if r['arm'] == arm]) for arm in replay['arms']}
    summary = {'run_id': run_id, 'task_ids': replay['task_ids'], 'record_ids': metadata['record_ids'],
        'metrics': metrics, 'contract_success_rate_delta': metrics['adapter']['task_success_rate']-metrics['base']['task_success_rate'],
        'heldout_accessed': False, 'factual_review_required': True,
        'interpretation': metadata['interpretation']}
    print('PAIRED_SUMMARY='+json.dumps(summary, separators=(',', ':')), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
