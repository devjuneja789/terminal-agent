"""GPU-only exact-context predictions; no trainer, optimizer or tool execution."""
from __future__ import annotations

import ast
import base64
import gc
import importlib.metadata
import json
import sys
import time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.definitions import validate_tool_arguments
from training.agenttune.corrective_training_data import verified_rows
from training.agenttune.prepare_sft import digest,read_rows
from training.agenttune.weighted_loss import token_weights


def emit(record: dict) -> None:
    encoded=base64.b64encode(json.dumps(record,ensure_ascii=False).encode()).decode()
    chunks=[encoded[i:i+12000] for i in range(0,len(encoded),12000)]
    for part,chunk in enumerate(chunks,1):
        print('CORRECTIVE_CONTEXT_CHUNK='+json.dumps({'kind':record['kind'],'record_id':record['record_id'],
            'part':part,'parts':len(chunks),'data':chunk},separators=(',',':')),flush=True)


def compare_action(text: str,gold: str) -> dict:
    expected=json.loads(gold)[0]
    result={'schema_valid':False,'tool_name_match':False,'exact_action_match':False,'expected_tool':expected['tool_name']}
    try:
        actions=json.loads(text)
        if not isinstance(actions,list) or len(actions)!=1 or not isinstance(actions[0],dict):
            raise ValueError('expected exactly one native action')
        action=actions[0]
        if set(action)!={'tool_name','parameters'}:raise ValueError('invalid action fields')
        if action['tool_name']=='finish':
            args=action['parameters']
            if not isinstance(args,dict) or set(args)!={'answer'} or not isinstance(args['answer'],str):
                raise ValueError('invalid finish arguments')
        else:validate_tool_arguments(action['tool_name'],action['parameters'])
        result.update(schema_valid=True,selected_tool=action['tool_name'],
            tool_name_match=action['tool_name']==expected['tool_name'],exact_action_match=action==expected)
        if expected['tool_name']=='write_file' and action['tool_name']=='write_file':
            content=action['parameters']['content'];gold_content=expected['parameters']['content']
            result.update(write_path_match=action['parameters']['path']==expected['parameters']['path'],
                write_content_exact=content==gold_content)
            try:
                module=ast.parse(content)
                compile(module,'<predicted-write>','exec')
                result.update(write_syntax_valid=True,defined_functions=sorted(n.name for n in module.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))))
            except SyntaxError as exc:result.update(write_syntax_valid=False,write_syntax_error=str(exc))
    except (ValueError,TypeError,KeyError) as exc:result['parse_error']=str(exc)
    return result


def aggregate(predictions: list[dict]) -> dict:
    output={}
    for arm in ('base','adapter'):
        rows=[r for r in predictions if r['arm']==arm]
        by_tool={}
        for tool in sorted({r['expected_tool'] for r in rows}):
            values=[r for r in rows if r['expected_tool']==tool]
            by_tool[tool]={'count':len(values),'tool_name_matches':sum(r['comparison']['tool_name_match'] for r in values),
                'exact_action_matches':sum(r['comparison']['exact_action_match'] for r in values),
                'schema_valid':sum(r['comparison']['schema_valid'] for r in values)}
        output[arm]={'prediction_count':len(rows),'schema_valid':sum(r['comparison']['schema_valid'] for r in rows),
            'tool_name_matches':sum(r['comparison']['tool_name_match'] for r in rows),
            'exact_action_matches':sum(r['comparison']['exact_action_match'] for r in rows),
            'teacher_forced_exact':sum(r['teacher_forced']['exact_sequence'] for r in rows),
            'mean_action_token_accuracy':sum(r['teacher_forced']['token_accuracy'] for r in rows)/len(rows),
            'mean_action_nll':sum(r['teacher_forced']['mean_nll'] for r in rows)/len(rows),
            'generated_tokens':sum(len(r['generated_token_ids']) for r in rows),
            'cap_hits':sum(r['hit_token_cap'] for r in rows),'by_expected_tool':by_tool}
    return output


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed

    if not torch.cuda.is_available():raise RuntimeError('HF CUDA required; CPU model inference disabled')
    config=json.loads((ROOT/'configs/corrective_contexts.json').read_text())
    source=json.loads((ROOT/'data/training_result.json').read_text())
    rows=verified_rows(read_rows(ROOT/'data/approved.jsonl'),read_rows(ROOT/'data/rows.jsonl'))
    if digest(source)!=config['source_sha256'] or digest(rows)!=config['rows_sha256'] or len(rows)!=15:
        raise ValueError('input provenance changed')
    for name,version in config['runtime'].items():
        if importlib.metadata.version(name).split('+')[0]!=version:raise RuntimeError('runtime mismatch: '+name)
    set_seed(config['seed'])
    tokenizer=AutoTokenizer.from_pretrained(config['model_id'],revision=config['model_revision'])
    saved=AutoTokenizer.from_pretrained(config['adapter_repo'],revision=config['adapter_revision'],subfolder=config['adapter_subfolder'])
    if (tokenizer.get_chat_template()!=saved.get_chat_template() or tokenizer.get_vocab()!=saved.get_vocab()
        or any(getattr(tokenizer,k)!=getattr(saved,k) for k in ('eos_token_id','bos_token_id','pad_token_id'))):
        raise ValueError('adapter/base tokenizer mismatch')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    if tokenizer.eos_token_id!=6 or tokenizer.convert_tokens_to_ids('<|END_RESPONSE|>')!=261001:
        raise ValueError('native closing token mismatch')
    cases=[]
    annotations={a['action_index']:a for a in source['training_audit']['weight_annotations']}
    masks={(m['source_example_id'],m['source_message_index']):m for m in source['training_audit']['target_mask_checks']}
    for index,row in enumerate(rows):
        prompt=tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,return_dict=True)['input_ids']
        full=tokenizer.apply_chat_template(row['prompt']+row['completion'],tokenize=True,return_dict=True)['input_ids']
        if (full[:len(prompt)]!=prompt or len(prompt)>config['max_prompt_tokens'] or len(full)>4096
            or len(prompt)+config['max_new_tokens']>config['max_context_tokens']):
            raise ValueError('context would be trimmed or label prefix changed')
        labels=[-100]*len(prompt)+full[len(prompt):]
        vector,audit=token_weights(tokenizer,row,full,labels,config['loss_weighting'])
        if config.get('objective')=='tool_choice_only':
            from training.agenttune.tool_choice_objective import choice_weights
            vector,audit=choice_weights(labels,audit)
        if config.get('objective')=='joint_first_choice_completion':
            from training.agenttune.joint_choice_objective import joint_weights
            vector,audit=joint_weights(labels,vector,audit,config['decision_share'])
        old=annotations[index];mask=masks[(row['source_example_id'],row['source_message_index'])]
        if (audit['target_token_ids']!=old['target_token_ids'] or audit['label_positions']!=old['label_positions']
            or audit['target_weights']!=old['target_weights'] or len(prompt)!=mask['masked_tokens']
            or len(full)-len(prompt)!=mask['target_tokens'] or labels[-2:]!=[261001,6]):
            raise ValueError('exact training-token/label/weight reproduction failed')
        cases.append({'kind':'context','record_id':f'context:{index:02d}','action_index':index,
            'source_split':'train-support','source_example_id':row['source_example_id'],'source_task_id':row['source_task_id'],
            'source_message_index':row['source_message_index'],'messages':row['prompt'],'gold_action':row['completion'][0]['content'],
            'input_ids':full,'labels':labels,'generation_prompt_ids':prompt,'weight_audit':audit,
            'training_tokens_labels_weights_reproduced':True})
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    metadata={'run_id':run_id,'config':config,'config_sha256':digest(config),'context_ids':[c['record_id'] for c in cases],
        'prediction_ids':[f'{arm}:{i:02d}' for arm in config['arms'] for i in range(15)],
        'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'cuda_runtime':torch.version.cuda,'packages':{n:importlib.metadata.version(n) for n in config['runtime']},
        'quantization':quant.to_dict(),'training_performed':False,'tool_execution_performed':False,'heldout_accessed':False}
    print('CORRECTIVE_CONTEXT_ENVIRONMENT='+json.dumps(metadata,separators=(',',':')),flush=True)
    for case in cases:emit(case)
    predictions=[];deadline=time.monotonic()+config['max_inference_seconds']
    for arm in config['arms']:
        set_seed(config['seed'])
        model=AutoModelForCausalLM.from_pretrained(config['model_id'],revision=config['model_revision'],dtype=torch.float16,
            device_map={'':0},quantization_config=quant,attn_implementation='eager')
        if arm=='adapter':model=PeftModel.from_pretrained(model,config['adapter_repo'],revision=config['adapter_revision'],subfolder=config['adapter_subfolder'])
        model.requires_grad_(False);model.eval()
        model.generation_config.eos_token_id=config['eos_token_ids'];model.generation_config.pad_token_id=tokenizer.pad_token_id
        for case in cases:
            if time.monotonic()>=deadline:raise TimeoutError('fixed diagnostic deadline exhausted')
            positions=case['weight_audit']['label_positions'];started=time.monotonic()
            with torch.inference_mode():
                inputs=torch.tensor([case['input_ids']],device='cuda')
                output=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),use_cache=False)
                scores=output.logits[0,[p-1 for p in positions],:].float()
                gold=torch.tensor([case['labels'][p] for p in positions],device='cuda')
                best=scores.argmax(-1);correct=(best==gold).tolist()
                nll=float((scores.logsumexp(-1)-scores.gather(1,gold[:,None]).squeeze(1)).mean())
                scopes={}
                for kind in ('tool_name','changed_code','json_closure'):
                    selected={p for s in case['weight_audit']['spans'] if s['kind']==kind for p in s['label_positions']}
                    values=[correct[i] for i,p in enumerate(positions) if p in selected]
                    scopes[kind]={'tokens':len(values),'top1_correct':sum(values),'exact':all(values) if values else None}
                first=next((i for i,v in enumerate(correct) if not v),None)
                teacher={'token_accuracy':sum(correct)/len(correct),'exact_sequence':all(correct),'mean_nll':nll,
                    'scopes':scopes,'gold_token_ids':gold.tolist(),'top1_token_ids':best.tolist(),
                    'first_divergence':None if first is None else {'label_position':positions[first],
                        'gold_token':tokenizer.decode([int(gold[first])]),'predicted_token':tokenizer.decode([int(best[first])])}}
                del inputs,output,scores,gold,best
                set_seed(config['seed'])
                prompt=torch.tensor([case['generation_prompt_ids']],device='cuda')
                generated=model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),do_sample=False,use_cache=True,
                    max_new_tokens=config['max_new_tokens'])
                ids=generated[0,prompt.shape[-1]:].tolist();text=tokenizer.decode(ids,skip_special_tokens=True).strip()
                del prompt,generated
            prediction={'kind':'prediction','record_id':f'{arm}:{case["action_index"]:02d}','arm':arm,
                'action_index':case['action_index'],'context_record_id':case['record_id'],'source_split':'train-support',
                'source_example_id':case['source_example_id'],'source_task_id':case['source_task_id'],
                'expected_tool':json.loads(case['gold_action'])[0]['tool_name'],'gold_action':case['gold_action'],
                'teacher_forced':teacher,'generated_text':text,'generated_token_ids':ids,
                'configured_stop_ids':list(model.generation_config.eos_token_id),'hit_token_cap':len(ids)==config['max_new_tokens'],
                'latency_seconds':time.monotonic()-started,'comparison':compare_action(text,case['gold_action']),
                'training_target_eligible':False}
            emit(prediction);predictions.append(prediction)
            print(f"{prediction['record_id']} expected={prediction['expected_tool']} exact={prediction['comparison']['exact_action_match']} valid={prediction['comparison']['schema_valid']}",flush=True)
        if any(p.grad is not None for p in model.parameters()):raise RuntimeError('unexpected gradient accumulation')
        del model;gc.collect();torch.cuda.empty_cache()
    print('CORRECTIVE_CONTEXT_SUMMARY='+json.dumps({'run_id':run_id,'context_count':15,'prediction_count':30,
        'metrics':aggregate(predictions),'training_performed':False,'tool_execution_performed':False,'heldout_accessed':False},separators=(',',':')),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
