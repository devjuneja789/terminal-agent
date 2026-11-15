"""GPU-only next-action predictions under matched live-source observations."""
from __future__ import annotations

import base64
import gc
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from training.agenttune.prepare_sft import digest,read_rows


def emit(record: dict) -> None:
    encoded=base64.b64encode(json.dumps(record,ensure_ascii=False).encode()).decode()
    chunks=[encoded[i:i+12000] for i in range(0,len(encoded),12000)]
    for part,data in enumerate(chunks,1):
        print('OBSERVATION_INTERVENTION_CHUNK='+json.dumps({'record_id':record['record_id'],
            'part':part,'parts':len(chunks),'data':data},separators=(',',':')),flush=True)


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('HF CUDA required; CPU model inference is disabled')
    config=json.loads((ROOT/'configs/observation_intervention.json').read_text())
    cases=read_rows(ROOT/'data/cases.jsonl')
    if len(cases)!=6 or digest(cases)!=config['cases_sha256'] or [c['case_id'] for c in cases]!=config['case_ids']:
        raise ValueError('matched case lineage changed')
    for name,version in config['runtime'].items():
        if importlib.metadata.version(name).split('+')[0]!=version:
            raise RuntimeError(f'runtime mismatch: {name}')
    tokenizer=AutoTokenizer.from_pretrained(config['model_id'],revision=config['model_revision'])
    saved=AutoTokenizer.from_pretrained(config['adapter_repo'],revision=config['adapter_revision'],subfolder=config['adapter_subfolder'])
    if (tokenizer.get_chat_template()!=saved.get_chat_template() or tokenizer.get_vocab()!=saved.get_vocab()
        or any(getattr(tokenizer,k)!=getattr(saved,k) for k in ('bos_token_id','eos_token_id','pad_token_id'))):
        raise RuntimeError('saved adapter tokenizer differs from base')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    identities=[arm+':'+c['case_id'] for arm in config['arms'] for c in cases]
    metadata={'run_id':run_id,'config':config,'config_sha256':digest(config),'record_ids':identities,
        'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'packages':{n:importlib.metadata.version(n) for n in config['runtime']},'quantization':quant.to_dict(),
        'effective_template_sha256':digest(tokenizer.get_chat_template()),
        'training_performed':False,'heldout_accessed':False,'tool_execution_performed':False}
    print('OBSERVATION_INTERVENTION_ENVIRONMENT='+json.dumps(metadata,separators=(',',':')),flush=True)
    deadline=time.monotonic()+config['max_inference_seconds']
    for arm in config['arms']:
        set_seed(config['seed'])
        model=AutoModelForCausalLM.from_pretrained(config['model_id'],revision=config['model_revision'],
            dtype=torch.float16,device_map={'':0},quantization_config=quant,attn_implementation='eager')
        if arm=='adapter':
            model=PeftModel.from_pretrained(model,config['adapter_repo'],revision=config['adapter_revision'],subfolder=config['adapter_subfolder'])
        model.requires_grad_(False)
        model.eval()
        model.generation_config.eos_token_id=tokenizer.eos_token_id
        model.generation_config.pad_token_id=tokenizer.pad_token_id
        for case in cases:
            if time.monotonic()>=deadline:raise TimeoutError('fixed inference budget exhausted')
            if digest(case['messages'])!=case['messages_sha256']:raise ValueError('prompt changed')
            set_seed(config['seed'])
            prompt=tokenizer.apply_chat_template(case['messages'],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
            length=prompt['input_ids'].shape[-1]
            generation=config['generation']
            if length>generation['max_prompt_tokens'] or length+generation['max_new_tokens']>generation['max_context_tokens']:
                raise ValueError('refusing to trim matched prompts')
            started=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**prompt,max_new_tokens=generation['max_new_tokens'],
                    do_sample=generation['do_sample'],use_cache=generation['use_cache'])
            ids=generated[0,length:].tolist()
            text=tokenizer.decode(ids,skip_special_tokens=True).strip()
            record={**case,'record_id':arm+':'+case['case_id'],'run_id':run_id,'arm':arm,'source_split':'train-support',
                'input_token_ids':prompt['input_ids'][0].tolist(),'generated_token_ids':ids,'generated_text':text,
                'prompt_tokens':length,'generated_tokens':len(ids),'latency_seconds':time.monotonic()-started,
                'hit_token_cap':len(ids)==generation['max_new_tokens'],'ended_with_eos':bool(ids and ids[-1]==tokenizer.eos_token_id),
                'matches_archived_adapter_output':digest(text)==case['archived_output_sha256'] if arm=='adapter' and case['condition']==config.get('control_condition','original') else None,
                'training_target_eligible':False}
            emit(record)
            print(f"{record['record_id']}: tokens={len(ids)} cap={record['hit_token_cap']} archived_match={record['matches_archived_adapter_output']}",flush=True)
            del prompt,generated
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print('OBSERVATION_INTERVENTION_SUMMARY='+json.dumps({'run_id':run_id,'record_ids':identities,'prediction_count':len(identities),
        'training_performed':False,'heldout_accessed':False,'tool_execution_performed':False},separators=(',',':')),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
