"""Pinned HF CUDA label/closing-token audit, with no optimizer or training call."""
from __future__ import annotations

import base64
import gc
import importlib.metadata
import json
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from training.agenttune.prepare_sft import digest,read_rows
from training.agenttune.weighted_loss import token_weights
from tools.definitions import validate_tool_arguments


def emit(record: dict) -> None:
    encoded=base64.b64encode(json.dumps(record,ensure_ascii=False,default=str).encode()).decode()
    chunks=[encoded[i:i+12000] for i in range(0,len(encoded),12000)]
    for part,data in enumerate(chunks,1):
        print('CORRECTIVE_PREFLIGHT_CHUNK='+json.dumps({'record_id':record['record_id'],'part':part,'parts':len(chunks),'data':data},separators=(',',':')),flush=True)


def inspect_labels(ids: list[int],labels: list[int],prompt: list[int],closings: list[int]) -> dict:
    active=[i for i,v in enumerate(labels) if v!=-100]
    if len(ids)!=len(labels) or not active:raise ValueError('missing or misaligned labels')
    first=active[0]
    result={'first_label_position':first,'prompt_tokens':len(prompt),'target_tokens':len(active),'sequence_tokens':len(ids),
        'prompt_equals_generation_prefix':ids[:first]==prompt,'all_prompt_labels_masked':labels[:len(prompt)]==[-100]*len(prompt),
        'contiguous_completion_labels':active==list(range(first,len(labels))),
        'target_labels_equal_input_ids':all(labels[i]==ids[i] for i in active),
        'both_closing_tokens_supervised':labels[-2:]==closings,
        'earlier_closing_tokens_masked':all(labels[i]==-100 for i,v in enumerate(ids[:first]) if v in closings),
        'label_positions':active}
    return result


def action_valid(text: str) -> dict:
    try:
        actions=json.loads(text)
        if not isinstance(actions,list) or len(actions)!=1 or not isinstance(actions[0],dict):raise ValueError('expected one action')
        action=actions[0]
        if set(action)!={'tool_name','parameters'}:raise ValueError('invalid action fields')
        if action['tool_name']=='finish':
            if set(action['parameters'])!={'answer'} or not isinstance(action['parameters']['answer'],str):raise ValueError('invalid finish')
        else:validate_tool_arguments(action['tool_name'],action['parameters'])
        return {'schema_valid':True,'selected_tool':action['tool_name']}
    except (ValueError,TypeError,KeyError) as exc:return {'schema_valid':False,'error':str(exc)}


def main() -> int:
    import torch
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    from peft import LoraConfig,PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
    from trl import SFTConfig,SFTTrainer

    if not torch.cuda.is_available():raise RuntimeError('HF CUDA required; CPU model inference is disabled')
    config=json.loads((ROOT/'configs/corrective_preflight.json').read_text())
    rows=read_rows(ROOT/'data/rows.jsonl');cases=read_rows(ROOT/'data/cases.jsonl')
    if len(rows)!=15 or digest(rows)!=config['rows_sha256'] or digest(cases)!=config['cases_sha256'] or len(cases)!=3:
        raise ValueError('preflight input lineage changed')
    for name,version in config['runtime'].items():
        if importlib.metadata.version(name).split('+')[0]!=version:raise RuntimeError('runtime mismatch: '+name)
    for row in rows:
        if not row['training_eligible'] or row['source_task_id'] not in ('B021','B026','B041','B046'):
            raise ValueError('unadmitted or non-support input')
    set_seed(config['seed'])
    tokenizer=AutoTokenizer.from_pretrained(config['model_id'],revision=config['model_revision'])
    saved=AutoTokenizer.from_pretrained(config['adapter_repo'],revision=config['adapter_revision'],subfolder=config['adapter_subfolder'])
    if (tokenizer.get_chat_template()!=saved.get_chat_template() or tokenizer.get_vocab()!=saved.get_vocab()
        or any(getattr(tokenizer,k)!=getattr(saved,k) for k in ('eos_token_id','bos_token_id','pad_token_id'))):
        raise RuntimeError('adapter tokenizer mismatch')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    response_end=tokenizer.convert_tokens_to_ids('<|END_RESPONSE|>');turn_end=tokenizer.eos_token_id
    if [response_end,turn_end]!=[261001,6]:raise RuntimeError('pinned closing tokens changed')
    upstream={}
    for name in ('config.json','generation_config.json','tokenizer_config.json'):
        path=hf_hub_download(config['model_id'],name,revision=config['model_revision'])
        value=json.loads(Path(path).read_text())
        upstream[name]={k:value[k] for k in ('eos_token_id','bos_token_id','pad_token_id','eos_token','bos_token','pad_token') if k in value}
        upstream[name]['file_sha256']=__import__('hashlib').sha256(Path(path).read_bytes()).hexdigest()
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    def load_base():
        return AutoModelForCausalLM.from_pretrained(config['model_id'],revision=config['model_revision'],dtype=torch.float16,
            device_map={'':0},quantization_config=quant,attn_implementation='eager')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    label_ids=[f"label:{r['source_example_id']}:{r['source_message_index']}" for r in rows]
    generation_ids=[f"generation:{arm}:{policy}:{c['task_id']}" for arm in config['arms'] for policy in config['stop_policies'] for c in cases]
    metadata={'run_id':run_id,'config':config,'config_sha256':digest(config),'record_ids':label_ids+generation_ids,
        'upstream_pinned_configs':upstream,'closing_ids':[response_end,turn_end],
        'gpu':torch.cuda.get_device_name(0),'packages':{n:importlib.metadata.version(n) for n in config['runtime']},
        'quantization':quant.to_dict(),'training_performed':False,'heldout_accessed':False,'tool_execution_performed':False}
    print('CORRECTIVE_PREFLIGHT_ENVIRONMENT='+json.dumps(metadata,separators=(',',':')),flush=True)

    temporary=load_base()
    args=SFTConfig(output_dir='/tmp/corrective-label-preflight',max_steps=1,per_device_train_batch_size=1,
        gradient_accumulation_steps=1,max_length=config['max_length'],completion_only_loss=True,
        fp16=True,bf16=False,packing=False,report_to=[],save_strategy='no',seed=config['seed'],
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    trainer=SFTTrainer(model=temporary,args=args,processing_class=tokenizer,
        train_dataset=Dataset.from_list([{'prompt':r['prompt'],'completion':r['completion']} for r in rows]),
        peft_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0,task_type='CAUSAL_LM',target_modules=['q_proj','v_proj']))
    if trainer.optimizer is not None or trainer.state.global_step!=0:raise RuntimeError('unexpected optimizer/update')
    audits=[]
    for dataset_index in range(len(trainer.train_dataset)):
        processed=trainer.train_dataset[dataset_index]
        matches=[i for i,r in enumerate(rows) if r['prompt']==processed['prompt'] and r['completion']==processed['completion']]
        if len(matches)!=1:raise ValueError('cannot match prepared action')
        index=matches[0];row=rows[index];batch=trainer.data_collator([processed])
        ids=batch['input_ids'][0].tolist();labels=batch['labels'][0].tolist()
        prompt_ids=tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,return_dict=True)['input_ids']
        audit=inspect_labels(ids,labels,prompt_ids,[response_end,turn_end])
        full=tokenizer.apply_chat_template(row['prompt']+row['completion'],tokenize=True,return_dict=True)['input_ids']
        decoded=tokenizer.decode([v for v in labels if v!=-100],skip_special_tokens=True)
        audit.update(full_template_equals_trainer_ids=full==ids,decoded_target_equals_reviewed_action=decoded==row['completion'][0]['content'],
            no_truncation=len(full)==len(ids) and len(full)<=config['max_length'])
        vector,weights=token_weights(tokenizer,row,ids,labels,config['loss_weighting'])
        audit['both_closing_token_weights_one']=vector[-2:]==[1.0,1.0]
        checks=[v for k,v in audit.items() if isinstance(v,bool)]
        if not all(checks):raise RuntimeError(f'label preflight failed for {label_ids[index]}: {audit}')
        record={'record_id':label_ids[index],'kind':'label','run_id':run_id,'source_split':'train-support',
            'source_example_id':row['source_example_id'],'source_task_id':row['source_task_id'],
            'source_message_index':row['source_message_index'],'context_message_count':row['context_message_count'],
            'input_ids':ids,'labels':labels,'generation_prompt_ids':prompt_ids,'gold_action':row['completion'][0]['content'],
            'audit':audit,'weight_audit':weights}
        emit(record);audits.append(record)
    if len(audits)!=15 or any(p.grad is not None for p in trainer.model.parameters()):raise RuntimeError('label audit incomplete or gradients accumulated')
    del trainer,temporary,batch
    gc.collect();torch.cuda.empty_cache()

    generations=[];deadline=time.monotonic()+config['max_inference_seconds']
    for arm in config['arms']:
        set_seed(config['seed']);model=load_base()
        if arm=='adapter':model=PeftModel.from_pretrained(model,config['adapter_repo'],revision=config['adapter_revision'],subfolder=config['adapter_subfolder'])
        model.requires_grad_(False);model.eval();model.generation_config.pad_token_id=tokenizer.pad_token_id
        for policy,stops in config['stop_policies'].items():
            model.generation_config.eos_token_id=stops
            for case in cases:
                if time.monotonic()>=deadline:raise TimeoutError('fixed inference deadline exhausted')
                if digest(case['messages'])!=case['messages_sha256']:raise ValueError('case prompt changed')
                set_seed(config['seed'])
                prompt=tokenizer.apply_chat_template(case['messages'],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
                length=prompt['input_ids'].shape[-1]
                if length>7000 or length+config['max_new_tokens']>8192:raise ValueError('refusing to trim generation case')
                started=time.monotonic()
                with torch.inference_mode():
                    generated=model.generate(**prompt,max_new_tokens=config['max_new_tokens'],do_sample=False,use_cache=True)
                ids=generated[0,length:].tolist();text=tokenizer.decode(ids,skip_special_tokens=True).strip()
                record={'record_id':f"generation:{arm}:{policy}:{case['task_id']}",'kind':'generation','run_id':run_id,
                    'source_split':'train-support','arm':arm,'policy':policy,'stop_ids':stops,'task_id':case['task_id'],
                    'messages':case['messages'],'input_token_ids':prompt['input_ids'][0].tolist(),'generated_token_ids':ids,
                    'generated_text':text,'generated_tokens':len(ids),'prompt_tokens':length,'latency_seconds':time.monotonic()-started,
                    'hit_token_cap':len(ids)==config['max_new_tokens'],'response_end_count':ids.count(response_end),
                    'last_token_id':ids[-1] if ids else None,'action':action_valid(text),
                    'matches_archived_legacy_output':digest(text)==case['legacy_output_sha256'][arm] if policy=='legacy_turn_only' else None,
                    'training_target_eligible':False}
                emit(record);generations.append(record)
                print(f"{record['record_id']}: tokens={len(ids)} cap={record['hit_token_cap']} valid={record['action']['schema_valid']}",flush=True)
                del prompt,generated
        del model;gc.collect();torch.cuda.empty_cache()
    summary={'run_id':run_id,'record_ids':metadata['record_ids'],'label_count':len(audits),'generation_count':len(generations),
        'all_label_checks_passed':True,'max_sequence_tokens':max(r['audit']['sequence_tokens'] for r in audits),
        'training_performed':False,'optimizer_created':False,'heldout_accessed':False,'tool_execution_performed':False}
    print('CORRECTIVE_PREFLIGHT_SUMMARY='+json.dumps(summary,separators=(',',':')),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
