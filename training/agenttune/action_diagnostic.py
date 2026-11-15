"""Inference-only inspection of approved action labels and predictions on HF CUDA."""
from __future__ import annotations

import base64
import gc
import importlib.metadata
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.definitions import validate_tool_arguments
from training.agenttune.prepare_sft import action_rows, digest, read_rows, verified_references
from training.agenttune.sft_smoke import normalize_trainable_precision


def emit(kind: str, identity: str, record: dict) -> None:
    encoded = base64.b64encode(json.dumps(record, ensure_ascii=False, default=str).encode()).decode()
    chunks = [encoded[i:i+12000] for i in range(0, len(encoded), 12000)]
    for part, chunk in enumerate(chunks, 1):
        print('ACTION_DIAGNOSTIC_CHUNK='+json.dumps({'kind': kind, 'record_id': identity,
            'part': part, 'parts': len(chunks), 'data': chunk}, separators=(',', ':')), flush=True)


def inspect_boundary(input_ids: list[int], labels: list[int], prompt_ids: list[int]) -> dict:
    if len(input_ids) != len(labels):
        raise ValueError('label/input lengths differ')
    active = [i for i, value in enumerate(labels) if value != -100]
    if not active:
        raise ValueError('no completion labels')
    start = active[0]
    return {'first_label_position': start, 'label_positions': active,
        'masked_tokens': len(labels)-len(active), 'target_tokens': len(active),
        'contiguous_completion_suffix': active == list(range(start, len(labels))),
        'target_labels_equal_input_tokens': all(labels[i] == input_ids[i] for i in active),
        'training_prefix_equals_generation_prompt': input_ids[:start] == prompt_ids,
        'generation_prompt_tokens': len(prompt_ids)}


def match_prepared_row(processed: dict, sources: list[dict]) -> int:
    matches = [i for i, row in enumerate(sources) if row['prompt'] == processed['prompt'] and row['completion'] == processed['completion']]
    if len(matches) != 1:
        raise ValueError('prepared row cannot be uniquely matched to reviewed messages')
    return matches[0]


def parse_generated(text: str, gold: str) -> dict:
    expected = json.loads(gold)[0]
    output = {'schema_valid': False, 'exact_action_match': False, 'tool_name_match': False}
    try:
        actions = json.loads(text)
        if not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], dict):
            raise ValueError('expected exactly one action')
        action = actions[0]
        if set(action) != {'tool_name', 'parameters'}:
            raise ValueError('invalid action fields')
        if action['tool_name'] == 'finish':
            if not isinstance(action['parameters'], dict) or set(action['parameters']) != {'answer'} or not isinstance(action['parameters']['answer'], str):
                raise ValueError('invalid finish arguments')
        else:
            validate_tool_arguments(action['tool_name'], action['parameters'])
        output.update(schema_valid=True, exact_action_match=action == expected,
            tool_name_match=action['tool_name'] == expected['tool_name'], selected_tool=action['tool_name'])
    except (ValueError, TypeError, KeyError) as exc:
        output['parse_error'] = str(exc)
    return output


def main() -> int:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError('diagnostic requires HF CUDA; CPU model inference is disabled')
    config = json.loads((ROOT/'configs/action_diagnostic.json').read_text())
    source = json.loads((ROOT/'data/training_result.json').read_text())
    profile = source['config']
    for name, version in config['runtime'].items():
        if importlib.metadata.version(name).split('+')[0] != version:
            raise RuntimeError(f'runtime mismatch: {name}')
    refs = verified_references(read_rows(ROOT/'data/source_records.jsonl'), read_rows(ROOT/'data/reviews.jsonl'), read_rows(ROOT/'data/approved.jsonl'))
    rows = action_rows(refs)
    if len(rows) != 16 or digest(rows) != source['rows_sha256']:
        raise RuntimeError('approved action-target lineage mismatch')
    set_seed(profile['seed'])
    tokenizer = AutoTokenizer.from_pretrained(profile['model_id'], revision=profile['model_revision'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    def load_base():
        return AutoModelForCausalLM.from_pretrained(profile['model_id'], revision=profile['model_revision'],
            dtype=torch.float16, device_map={'': 0}, quantization_config=quant, attn_implementation='eager')

    # Reproduce the exact trainer preprocessing; never create an optimizer or call train/backward.
    temporary_model = load_base()
    args = SFTConfig(output_dir='/tmp/label-inspection', max_steps=64, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=profile['learning_rate'], max_length=profile['max_length'],
        completion_only_loss=True, fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, logging_steps=1,
        save_strategy='no', report_to=[], seed=profile['seed'], optim='adamw_torch', packing=False)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0, revision=profile['model_revision'],
                     task_type='CAUSAL_LM', target_modules=['q_proj','v_proj'])
    trainer = SFTTrainer(model=temporary_model, args=args, processing_class=tokenizer,
        train_dataset=Dataset.from_list([{'prompt': r['prompt'], 'completion': r['completion']} for r in rows]),
        peft_config=lora)
    normalize_trainable_precision(trainer.model)
    if trainer.optimizer is not None or trainer.state.global_step != 0:
        raise RuntimeError('diagnostic unexpectedly constructed optimizer state')
    prepared = {}
    mismatched_audit_ids = 0
    for dataset_index in range(len(trainer.train_dataset)):
        processed = trainer.train_dataset[dataset_index]
        index = match_prepared_row(processed, rows)
        row = rows[index]
        batch = trainer.data_collator([processed])
        ids, labels = batch['input_ids'][0].tolist(), batch['labels'][0].tolist()
        prompt_ids = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True, return_dict=True)['input_ids']
        boundary = inspect_boundary(ids, labels, prompt_ids)
        gold = row['completion'][0]['content']
        gold_tool = json.loads(gold)[0]['tool_name']
        rendered = tokenizer.apply_chat_template(row['prompt']+row['completion'], tokenize=False)
        full_ids = tokenizer.apply_chat_template(row['prompt']+row['completion'], tokenize=True, return_dict=True)['input_ids']
        offset_check = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        completion_start = rendered.rfind(gold)
        tool_match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', gold)
        tool_positions = []
        if offset_check['input_ids'] == ids and completion_start >= 0 and tool_match:
            left, right = (completion_start+v for v in tool_match.span(1))
            tool_positions = [i for i,(a,b) in enumerate(offset_check['offset_mapping'])
                              if a < right and b > left and i in boundary['label_positions']]
        old = source['training_audit']['target_mask_checks'][dataset_index]
        mismatched_audit_ids += old['source_example_id'] != row['source_example_id']
        identity = f"{row['source_example_id']}:action-{index:02d}"
        record = {'record_id': identity, 'source_split': 'train-support', 'source_example_id': row['source_example_id'],
            'source_task_id': row['source_task_id'], 'action_index': index, 'dataset_index': dataset_index,
            'expected_tool': gold_tool, 'gold_action': gold, 'input_ids': ids, 'labels': labels,
            'generation_prompt_ids': prompt_ids, 'rendered_training_text': rendered,
            'decoded_labels': tokenizer.decode([labels[i] for i in boundary['label_positions']], skip_special_tokens=False),
            'decoded_label_content': tokenizer.decode([labels[i] for i in boundary['label_positions']], skip_special_tokens=True),
            'training_full_sequence_matches_template': full_ids == ids,
            'offset_mapping_matches_training_sequence': offset_check['input_ids'] == ids,
            'tool_name_label_positions': tool_positions,
            'end_of_turn_supervised': labels[-1] == tokenizer.eos_token_id,
            'eos_label_count': sum(labels[i] == tokenizer.eos_token_id for i in boundary['label_positions']),
            'previous_audit_source_id': old['source_example_id'], **boundary}
        prepared[index] = record
        emit('boundary', identity, record)
    if len(prepared) != 16:
        raise RuntimeError('not all action targets recovered')
    old_counts = Counter((r['masked_tokens'],r['target_tokens']) for r in source['training_audit']['target_mask_checks'])
    new_counts = Counter((r['masked_tokens'],r['target_tokens']) for r in prepared.values())
    boundary_summary = {'action_count':16, 'previous_audit_count_multiset_reproduced':old_counts == new_counts,
        'previous_audit_source_ids_mismatched_due_to_shuffle': mismatched_audit_ids,
        'prefix_matches':sum(r['training_prefix_equals_generation_prompt'] for r in prepared.values()),
        'exact_gold_label_content_matches':sum(r['decoded_label_content'] == r['gold_action'] for r in prepared.values()),
        'eos_supervised':sum(r['end_of_turn_supervised'] for r in prepared.values()),
        'tool_name_spans_identified':sum(bool(r['tool_name_label_positions']) for r in prepared.values())}
    del trainer, temporary_model, batch
    gc.collect()
    torch.cuda.empty_cache()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    metadata = {'run_id':run_id, 'run_kind':'approved_action_label_diagnostic', 'config':config,
        'config_sha256':digest(config), 'rows_sha256':digest(rows), 'boundary_summary':boundary_summary,
        'source_training_run_id':source['run_id'], 'adapter_commit':source['artifact_commit'],
        'gpu':torch.cuda.get_device_name(0), 'packages':{n:importlib.metadata.version(n) for n in config['runtime']},
        'training_performed':False, 'tool_execution_performed':False, 'heldout_accessed':False,
        'boundary_ids':[prepared[i]['record_id'] for i in range(16)],
        'record_ids':[f'{arm}:{prepared[i]["record_id"]}' for arm in ('base','adapter') for i in range(16)]}
    print('ACTION_DIAGNOSTIC_ENVIRONMENT='+json.dumps(metadata, separators=(',',':')), flush=True)
    predictions = []
    deadline = time.monotonic()+config['max_inference_seconds']
    for arm in ('base','adapter'):
        model = load_base()
        if arm == 'adapter':
            model = PeftModel.from_pretrained(model, profile['artifact_repo'], revision=source['artifact_commit'], subfolder=source['run_id']+'/adapter')
        model.requires_grad_(False)
        model.eval()
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        for index in range(16):
            if time.monotonic() >= deadline:
                raise TimeoutError('fixed diagnostic inference budget exhausted')
            item, row = prepared[index], rows[index]
            positions = item['label_positions']
            inputs = torch.tensor([item['input_ids']], device='cuda')
            with torch.inference_mode():
                outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
                scores = outputs.logits[0, [p-1 for p in positions], :].float()
                gold_ids = torch.tensor([item['labels'][p] for p in positions], device='cuda')
                best = scores.argmax(-1)
                log_probs = scores.gather(1,gold_ids[:,None]).squeeze(1)-scores.logsumexp(-1)
                correct = (best == gold_ids).tolist()
                details = []
                first_wrong = next((i for i,value in enumerate(correct) if not value), None)
                for local, absolute in enumerate(positions):
                    if correct[local] and absolute not in item['tool_name_label_positions']:
                        continue
                    top = scores[local].topk(5)
                    norm = scores[local].logsumexp(0)
                    details.append({'label_position':absolute, 'gold_token_id':int(gold_ids[local]),
                        'gold_token':tokenizer.decode([int(gold_ids[local])]),
                        'top1_token_id':int(best[local]), 'top1_token':tokenizer.decode([int(best[local])]),
                        'gold_probability':float(log_probs[local].exp()),
                        'gold_rank':int((scores[local] > scores[local,gold_ids[local]]).sum())+1,
                        'tool_name_token':absolute in item['tool_name_label_positions'],
                        'first_divergence':local == first_wrong,
                        'top5':[{'token_id':int(k),'token':tokenizer.decode([int(k)]),'probability':float((v-norm).exp())} for k,v in zip(top.indices,top.values)]})
                nll = float(-log_probs.mean())
                teacher = {'token_accuracy':sum(correct)/len(correct), 'exact_sequence':all(correct),
                    'mean_nll':nll, 'eos_top1_correct':bool(correct[-1]) if item['end_of_turn_supervised'] else None,
                    'tool_name_exact':all(correct[positions.index(p)] for p in item['tool_name_label_positions']) if item['tool_name_label_positions'] else None,
                    'mismatch_and_critical_tokens':details}
                del outputs, scores, best, log_probs, gold_ids, inputs
                set_seed(profile['seed'])
                prompt = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors='pt').to('cuda')
                generated = model.generate(**prompt, do_sample=False, use_cache=True,
                    max_new_tokens=config['max_new_tokens'], max_time=min(60,max(1,deadline-time.monotonic())))
                new_ids = generated[0,prompt['input_ids'].shape[1]:].tolist()
                text = tokenizer.decode(new_ids,skip_special_tokens=True).strip()
                del prompt, generated
            record = {'record_id':f'{arm}:{item["record_id"]}', 'arm':arm, 'source_split':'train-support',
                'source_example_id':row['source_example_id'], 'source_task_id':row['source_task_id'],
                'action_index':index, 'expected_tool':item['expected_tool'], 'gold_action':item['gold_action'],
                'teacher_forced':teacher, 'generated_text':text, 'generated_token_ids':new_ids,
                'ended_with_eos':bool(new_ids and new_ids[-1] == tokenizer.eos_token_id),
                'hit_token_cap':len(new_ids) == config['max_new_tokens'],
                'free_generation':parse_generated(text,item['gold_action'])}
            predictions.append(record)
            emit('prediction',record['record_id'],record)
            print(f"{arm} action {index:02d} {item['expected_tool']}: teacher_accuracy={teacher['token_accuracy']:.3f} free_exact={record['free_generation']['exact_action_match']}",flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    metrics = {}
    for arm in ('base','adapter'):
        values = [r for r in predictions if r['arm']==arm]
        metrics[arm] = {'actions':len(values),
            'teacher_forced_exact':sum(r['teacher_forced']['exact_sequence'] for r in values),
            'teacher_forced_tool_name_exact':sum(r['teacher_forced']['tool_name_exact'] is True for r in values),
            'teacher_forced_tool_name_scored':sum(r['teacher_forced']['tool_name_exact'] is not None for r in values),
            'mean_action_token_accuracy':sum(r['teacher_forced']['token_accuracy'] for r in values)/len(values),
            'mean_action_nll':sum(r['teacher_forced']['mean_nll'] for r in values)/len(values),
            'free_action_exact':sum(r['free_generation']['exact_action_match'] for r in values),
            'free_tool_name_match':sum(r['free_generation']['tool_name_match'] for r in values),
            'free_schema_valid':sum(r['free_generation']['schema_valid'] for r in values)}
    print('ACTION_DIAGNOSTIC_SUMMARY='+json.dumps({'run_id':run_id,'boundary_summary':boundary_summary,'metrics':metrics,
        'prediction_count':len(predictions),'training_performed':False,'tool_execution_performed':False,
        'heldout_accessed':False},separators=(',',':')),flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
