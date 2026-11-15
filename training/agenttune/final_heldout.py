"""Final GPU-only paired evaluation: no training, reward scoring or selection."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evals.benchmark.loader import BENCHMARK_ROOT,load_benchmark
from tools.definitions import get_tool_schemas
from training.agenttune.evaluate_heldout import _run_one,_aggregate,_hardware
from training.agenttune.hf_gpu_smoke import EVAL_CONFIG,MODEL_ID,MODEL_REVISION,verify_frozen_config,verify_runtime_versions,_canonical_sha256
from training.agenttune.paired_replay import verify_tokenizers,generation_stops,capture_generations,emit_record


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,GenerationConfig,set_seed
    if not torch.cuda.is_available():raise RuntimeError('HF CUDA required; CPU inference disabled')
    schemas=get_tool_schemas();verify_frozen_config(schemas);verify_runtime_versions(torch)
    config=EVAL_CONFIG;paired=config['paired']
    for name,version in paired['extra_runtime'].items():
        if importlib.metadata.version(name)!=version:raise RuntimeError('runtime mismatch: '+name)
    freeze=ROOT/'configs/submission_candidate.json'
    if hashlib.sha256(freeze.read_bytes()).hexdigest()!=config['candidate_freeze_sha256']:raise ValueError('candidate freeze changed')
    if hashlib.sha256((BENCHMARK_ROOT/'heldout_eval.jsonl').read_bytes()).hexdigest()!=config['heldout_taskfile_sha256']:
        raise ValueError('held-out task file changed')
    manifest=json.loads((BENCHMARK_ROOT/'manifest.json').read_text())
    if set(manifest['splits'])!={'held-out-eval'} or (BENCHMARK_ROOT/'train_support.jsonl').exists() or (ROOT/'data').exists():
        raise ValueError('final evaluation must contain only held-out tasks')
    tasks=list(load_benchmark('held-out-eval'))
    if [t.task_id for t in tasks]!=[f'B{i:03d}' for i in range(5,51,5)]:raise ValueError('held-out split changed')
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
    saved=AutoTokenizer.from_pretrained(paired['adapter_repo'],revision=paired['adapter_revision'],subfolder=paired['adapter_subfolder'])
    token_check=verify_tokenizers(tokenizer,saved)
    upstream=GenerationConfig.from_pretrained(MODEL_ID,revision=MODEL_REVISION).eos_token_id
    stops=generation_stops(paired,tokenizer,upstream)
    quant_args=dict(paired['quantization']);quant_args['bnb_4bit_compute_dtype']=torch.float16
    quant=BitsAndBytesConfig(**quant_args)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    metadata={'run_kind':'final_paired_heldout','run_id':run_id,'config':config,'config_sha256':_canonical_sha256(config),
        'task_ids':paired['task_ids'],'record_ids':[f'{a}:{t.task_id}' for a in paired['arms'] for t in tasks],
        'hardware':_hardware(torch),'tokenizer_equivalence':token_check,'quantization':quant.to_dict(),
        'generation_stop_ids':stops,'upstream_generation_eos_token_ids':upstream,
        'candidate_freeze_sha256':config['candidate_freeze_sha256'],'heldout_accessed':True,
        'training_performed':False,'reward_generation_performed':False,'model_selection_performed':False,
        'interpretation':'Fresh matched held-out comparison; contract success is deterministic benchmark success, not unrestricted coding reliability.'}
    print('PAIRED_ENVIRONMENT='+json.dumps(metadata,separators=(',',':')),flush=True)
    records=[]
    for arm in paired['arms']:
        set_seed(paired['seed'])
        model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REVISION,dtype=torch.float16,
            device_map={'':0},quantization_config=quant,attn_implementation='eager')
        if arm=='adapter':model=PeftModel.from_pretrained(model,paired['adapter_repo'],revision=paired['adapter_revision'],subfolder=paired['adapter_subfolder'])
        model.eval();model.generation_config.eos_token_id=stops;model.generation_config.pad_token_id=tokenizer.pad_token_id
        audit=capture_generations(model,config['generation']['max_new_tokens'])
        for task in tasks:
            set_seed(paired['seed']);start=len(audit)
            record,_=_run_one(task,tokenizer,model,torch,schemas,run_id=run_id,metadata=metadata,capture_support_audit=False)
            generations=audit[start:]
            for g in generations:g['decoded_text']=tokenizer.decode(g['generated_token_ids'],skip_special_tokens=True).strip()
            record.update(run_kind='final_paired_heldout',arm=arm,record_id=f'{arm}:{task.task_id}',
                training_target_eligible=False,training_use_prohibited=True,generation_audit=generations)
            # Nonzero process exit is evidence; actual executor faults have an error.
            if record['failure_category']=='tool_execution':
                faults=any(not e.get('schema_valid',True) or not isinstance(e.get('result'),dict)
                    or e['result'].get('error') is not None for e in record['agenttune_trace']['tool_executions'])
                if not faults:record.update(original_failure_category='tool_execution',failure_category='task_contract_failure')
            records.append(record);emit_record(record)
            print(f"{arm} {task.task_id}: {'PASS' if record['success'] else 'FAIL'} ({record['failure_category']})",flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    metrics={a:_aggregate([r for r in records if r['arm']==a]) for a in paired['arms']}
    summary={'run_id':run_id,'task_ids':paired['task_ids'],'record_ids':metadata['record_ids'],'metrics':metrics,
        'contract_success_rate_delta':metrics['adapter']['task_success_rate']-metrics['base']['task_success_rate'],
        'heldout_accessed':True,'training_performed':False,'factual_review_required':True,'interpretation':metadata['interpretation']}
    print('PAIRED_SUMMARY='+json.dumps(summary,separators=(',',':')),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
