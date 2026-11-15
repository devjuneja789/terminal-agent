"""Approved-reference AgentTune/TRL QLoRA; run only in an HF CUDA Job."""
from __future__ import annotations

import gc
import base64
import importlib.metadata
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.agenttune.prepare_sft import action_rows, digest, read_rows, verified_references
from training.agenttune.weighted_loss import token_weights, weighted_trainer_class, weighted_collator, initialize_loss_scale


def normalize_trainable_precision(model) -> dict:
    """TRL 1.7.1 casts QLoRA adapters to BF16; FP16 AMP needs FP32 grads.

    Call after SFTTrainer initialization and before optimizer creation. Never
    cast the frozen quantized base or an already accumulated gradient.
    """
    before = {}
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if 'lora_' not in name or not parameter.is_floating_point():
            raise RuntimeError(f'unexpected trainable parameter: {name}')
        if parameter.grad is not None:
            raise RuntimeError('precision normalization must precede backward')
        dtype = str(parameter.dtype)
        before[dtype] = before.get(dtype, 0) + parameter.numel()
        parameter.data = parameter.data.float()
        if str(parameter.dtype) != 'torch.float32':
            raise RuntimeError(f'could not normalize adapter dtype: {name}')
        count += parameter.numel()
    if not count:
        raise RuntimeError('no trainable LoRA weights')
    return {'before': before, 'after': {'torch.float32': count}}


def prepare_artifact_repo(api, repo: str) -> None:
    """Reuse only an empty private destination left by an unsuccessful smoke."""
    api.create_repo(repo_id=repo, repo_type='model', private=True, exist_ok=True)
    if not api.repo_info(repo_id=repo, repo_type='model').private:
        raise ValueError('artifact destination must be private')
    if set(api.list_repo_files(repo_id=repo, repo_type='model')) - {'.gitattributes'}:
        raise ValueError('artifact destination is populated; select a new repository')


def chat_token_length(tokenizer, messages: list[dict]) -> int:
    # Transformers 5 may return a BatchEncoding whose len() counts fields.
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)
    return len(encoded['input_ids'])


def verify_edit_target_coverage(annotations: list[dict], *, mixed: bool=False) -> None:
    expected=6 if mixed else 3
    edits=[a for a in annotations if a['expected_tool']=='write_file']
    if len(edits)!=expected or any(not any(s['kind']=='changed_code' for s in a['spans']) for a in edits):
        raise RuntimeError('weighted edit-target coverage changed')


def main() -> int:
    import torch
    from agenttune.agentic import Event, EventKind, EventLog, Project
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU training/inference is disabled')
    config = json.loads((ROOT/'configs/tiny_aya_sft_smoke.json').read_text())
    for name, version in config.get('extra_runtime', {}).items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'runtime mismatch for {name}')
    set_seed(config['seed'])
    repo = config['artifact_repo']
    if not repo:
        raise ValueError('a new private artifact repository must be supplied before launch')
    mixed = config.get('dataset_kind') == 'approved_task_start_plus_corrections_v1'
    corrective = mixed or config.get('dataset_kind') == 'admitted_corrective_continuations_v1'
    choice_only = config.get('objective') == 'tool_choice_only'
    joint = config.get('objective') == 'joint_first_choice_completion'
    if (choice_only or joint) and not corrective:
        raise ValueError('tool-choice experiment requires admitted corrective data')
    if corrective:
        if mixed:
            from training.agenttune.task_start_data import verified_rows
        else:
            from training.agenttune.corrective_training_data import verified_rows
        refs = read_rows(ROOT/'data/approved.jsonl')
        expected_rows = verified_rows(refs, read_rows(ROOT/'data/corrective_rows.jsonl'))
        manifest = json.loads((ROOT/'data/manifest.json').read_text())
        if (digest(refs) != manifest['refs_sha256'] or digest(config) != manifest['config_sha256']
            or digest(expected_rows) != config['rows_sha256'] or config['eos_token_ids'] != [3,6,261001]
            or (not mixed and not manifest['label_gate']['all_label_checks_passed'])
            or not manifest['runtime_gate']['runtime_stop_gate_passed']):
            raise ValueError('corrective training lineage or passed gates changed')
    else:
        refs = verified_references(read_rows(ROOT/'data/source_records.jsonl'),
                                   read_rows(ROOT/'data/reviews.jsonl'), read_rows(ROOT/'data/approved.jsonl'))
        expected_rows = action_rows(refs)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path('/tmp/terminal-slm-sft-output')/run_id
    output.mkdir(parents=True)
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    prepare_artifact_repo(api, repo)
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            config['model_id'], revision=config['model_revision'], quantization_config=quant,
            dtype=torch.float16, device_map={'': 0}, attn_implementation='eager')

    project = Project()
    for ref in refs:
        log = EventLog(tier='full')
        # Use TEXT for JSON actions: TOOL_CALL exports would rewrite the protocol.
        # These are reviewed demonstrations, with no fabricated policy logprobs.
        for message in ref['messages'][1:]:
            kind = EventKind.TEXT if message['role'] == 'assistant' else EventKind.OBSERVATION
            log.append(Event(kind, {'text': message['content']}))
        project.add_trajectory(log)

    holder = {}
    class StepAudit(TrainerCallback):
        accelerator = None
        skipped_steps = []

        def on_step_end(self, args, state, control, **kwargs):
            if self.accelerator.optimizer_step_was_skipped:
                self.skipped_steps.append(state.global_step)
                if 'loss_weighting' in config:
                    raise RuntimeError(f'weighted run skipped optimizer update {state.global_step}; aborting')

    def factory(*, train_dataset):
        rebuilt = []
        for ref, row in zip(refs, train_dataset, strict=True):
            messages = [ref['messages'][0], *row['messages']]
            if messages != ref['messages']:
                raise RuntimeError('AgentTune export changed the reviewed conversation')
            rebuilt.append({**ref, 'messages': messages})
        rows = verified_rows(rebuilt, expected_rows) if corrective else action_rows(rebuilt)
        if rows != expected_rows:
            raise RuntimeError('AgentTune targets differ from prepared targets')
        lengths = [chat_token_length(tokenizer, row['prompt']+row['completion']) for row in rows]
        if max(lengths) > config['max_length']:
            raise RuntimeError(f"refusing to truncate reviewed examples: max tokens={max(lengths)}")
        model = load_base()
        model.config.use_cache = False
        args = SFTConfig(output_dir=str(output/'training'), max_steps=config['max_steps'],
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            learning_rate=config['learning_rate'], max_length=config['max_length'],
            lr_scheduler_type=config.get('lr_scheduler_type', 'linear'),
            warmup_steps=config.get('warmup_steps', 0), max_grad_norm=config.get('max_grad_norm', 1.0),
            completion_only_loss=True, fp16=True, bf16=False, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            logging_steps=1, save_strategy='no', report_to=[], seed=config['seed'],
            optim='adamw_torch', packing=False)
        lora = LoraConfig(r=config['lora_rank'], lora_alpha=16, lora_dropout=0, revision=config['model_revision'],
                          task_type='CAUSAL_LM', target_modules=['q_proj', 'v_proj'])
        trainer_type = weighted_trainer_class(SFTTrainer,tool_choice_only=choice_only) if 'loss_weighting' in config else SFTTrainer
        trainer = trainer_type(model=model, args=args, processing_class=tokenizer,
            train_dataset=Dataset.from_list([{'prompt': r['prompt'], 'completion': r['completion']} for r in rows]),
            peft_config=lora)
        precision = normalize_trainable_precision(trainer.model)
        if corrective:
            trainer.model.generation_config.eos_token_id = config['eos_token_ids']
        scaler_audit = initialize_loss_scale(trainer.accelerator,config['amp_initial_scale']) if 'loss_weighting' in config else None
        print('SFT_TRAINABLE_PRECISION='+json.dumps(precision), flush=True)
        masks = []
        annotations = []
        weight_vectors = []
        for index in range(len(trainer.train_dataset)):
            processed = trainer.train_dataset[index]
            matches = [i for i,row in enumerate(rows) if row['prompt']==processed['prompt'] and row['completion']==processed['completion']]
            if len(matches)!=1:
                raise RuntimeError('prepared action cannot be matched to its reviewed source')
            source_index = matches[0]
            batch = trainer.data_collator([processed])
            masked = int((batch['labels'] == -100).sum())
            target = int((batch['labels'] != -100).sum())
            if masked == 0 or target == 0:
                raise RuntimeError(f'completion masking invalid for action target {index}')
            label_audit = {}
            if corrective:
                from training.agenttune.corrective_preflight import inspect_labels
                ids, labels = batch['input_ids'][0].tolist(), batch['labels'][0].tolist()
                prompt_ids = tokenizer.apply_chat_template(rows[source_index]['prompt'], tokenize=True,
                    add_generation_prompt=True, return_dict=True)['input_ids']
                label_audit = inspect_labels(ids, labels, prompt_ids, [261001,6])
                label_audit.update(decoded_target_equals_reviewed_action=
                    tokenizer.decode([v for v in labels if v != -100], skip_special_tokens=True) == rows[source_index]['completion'][0]['content'],
                    full_template_equals_trainer_ids=ids == tokenizer.apply_chat_template(
                        rows[source_index]['prompt']+rows[source_index]['completion'], tokenize=True, return_dict=True)['input_ids'])
                if not all(v for v in label_audit.values() if isinstance(v,bool)):
                    raise RuntimeError('corrective completion label boundary changed')
            masks.append({'source_example_id': rows[source_index]['source_example_id'],
                          'masked_tokens': masked, 'target_tokens': target,
                          **({'source_message_index':rows[source_index].get('source_message_index',len(rows[source_index]['prompt'])), 'label_audit':label_audit} if corrective else {})})
            if 'loss_weighting' in config:
                vector,annotation = token_weights(tokenizer,rows[source_index],
                    batch['input_ids'][0].tolist(),batch['labels'][0].tolist(),config['loss_weighting'])
                annotation.update(action_index=source_index,dataset_index=index)
                if choice_only:
                    from training.agenttune.tool_choice_objective import choice_weights
                    vector,annotation = choice_weights(batch['labels'][0].tolist(),annotation)
                    masks[-1]['effective_supervision']={'objective':'tool_choice_only',
                        'positive_loss_tokens':len(annotation['selected_tool_positions']),
                        'arguments_supervised':False,'native_closings_supervised':False,
                        'full_label_audit_describes_retained_template_not_loss_coefficients':True}
                if joint:
                    from training.agenttune.joint_choice_objective import joint_weights
                    vector,annotation=joint_weights(batch['labels'][0].tolist(),vector,annotation,config['decision_share'])
                    masks[-1]['effective_supervision']={'objective':'joint_first_choice_completion',
                        'all_completion_tokens_supervised':True,'native_closings_supervised':True,
                        'decision_share':config['decision_share']}
                if corrective and (any(v<=0 for v in vector[-2:]) if joint else vector[-2:] != ([0.0,0.0] if choice_only else [1.0,1.0])):
                    raise RuntimeError('native closing token weights changed')
                weight_vectors.append(vector)
                annotations.append(annotation)
        if 'loss_weighting' in config:
            trainer.train_dataset = trainer.train_dataset.add_column('loss_weights',weight_vectors)
            trainer.data_collator = weighted_collator(trainer.data_collator)
            trainer.model_accepts_loss_kwargs = False
            trainer._signature_columns = None
            trainer._set_signature_columns_if_needed()
            verify_edit_target_coverage(annotations,mixed=mixed)
        audit = StepAudit()
        audit.skipped_steps = []
        audit.accelerator = trainer.accelerator
        trainer.add_callback(audit)
        objective_check=None
        if choice_only:
            from training.agenttune.tool_choice_objective import loss_self_check
            objective_check=loss_self_check(torch)
            if config.get('gradient_preflight_all_batches'):
                from training.agenttune.tool_choice_objective import gradient_preflight
                objective_check['gradient_preflight']=gradient_preflight(trainer,torch,config['amp_initial_scale'])
                print('SFT_CHOICE_GRADIENT_PREFLIGHT='+json.dumps(objective_check['gradient_preflight']),flush=True)
        if joint:
            from training.agenttune.joint_choice_objective import loss_self_check
            from training.agenttune.tool_choice_objective import gradient_preflight
            objective_check=loss_self_check(torch)
            objective_check['gradient_preflight']=gradient_preflight(trainer,torch,config['amp_initial_scale'])
            print('SFT_JOINT_GRADIENT_PREFLIGHT='+json.dumps(objective_check['gradient_preflight']),flush=True)
        holder.update(trainer=trainer, before={n: p.detach().cpu().clone() for n,p in trainer.model.named_parameters() if 'lora_B' in n}, max_tokens=max(lengths), precision=precision, masks=masks, step_audit=audit, annotations=annotations, scaler_audit=scaler_audit, objective_check=objective_check)
        return trainer

    result = project.train(trainer_factory=factory, fmt='sft')
    trainer = holder['trainer']
    changed = any(not torch.equal(holder['before'][n], p.detach().cpu())
                  for n,p in trainer.model.named_parameters() if n in holder['before'])
    if not changed or not math.isfinite(result.training_loss):
        raise RuntimeError('no adapter update or nonfinite training loss')
    if trainer.state.global_step != config['max_steps'] or holder['step_audit'].skipped_steps:
        raise RuntimeError('training stopped early or skipped optimizer updates')
    if trainer.state.epoch < config.get('expected_epochs', 0) - 0.001:
        raise RuntimeError('training did not complete the declared epochs')
    adapter_path = output/'adapter'
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    precision = holder['precision']
    training_audit = {'global_step': trainer.state.global_step, 'epoch': trainer.state.epoch,
        'skipped_optimizer_steps': holder['step_audit'].skipped_steps,
        'target_mask_checks': holder['masks'], 'max_training_tokens': holder['max_tokens'],
        'log_history': trainer.state.log_history}
    if choice_only:
        training_audit['tool_choice_objective_self_check']=holder['objective_check']
    if joint:
        training_audit['joint_objective_self_check']=holder['objective_check']
    if 'loss_weighting' in config:
        training_audit['weight_annotations'] = holder['annotations']
        training_audit['amp_scaler_initialization'] = holder['scaler_audit']
        (output/'loss-weight-audit.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in holder['annotations']))
    (output/'training-history.jsonl').write_text(''.join(json.dumps(row, default=str)+'\n' for row in trainer.state.log_history))
    holder.clear()
    del trainer, project
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load_base(), adapter_path).eval()
    if corrective:
        reloaded.generation_config.eos_token_id = config['eos_token_ids']
        reloaded.generation_config.pad_token_id = tokenizer.pad_token_id
    probe = tokenizer.apply_chat_template(expected_rows[0]['prompt'], add_generation_prompt=True,
                                         return_tensors='pt', return_dict=True).to('cuda')
    with torch.inference_mode():
        generated = reloaded.generate(**probe, max_new_tokens=64, do_sample=False)
    probe_text = tokenizer.decode(generated[0, probe['input_ids'].shape[1]:], skip_special_tokens=True)
    metadata = {'run_kind': config.get('experiment_kind', 'sft_pipeline_smoke'), 'config': config, 'run_id': run_id,
        'source_examples': [r['example_id'] for r in refs], 'rows_sha256': digest(expected_rows),
        'trajectory_count': len(refs), 'action_target_count': len(expected_rows),
        'training_metrics': result.metrics, 'adapter_changed': changed, 'adapter_reload_passed': True,
        'quantization': quant.to_dict(), 'loss': 'joint_first_choice_completion' if joint else ('tool_choice_only' if choice_only else ('decision_edit_weighted_completion_only' if 'loss_weighting' in config else 'completion_only')),
        'trainable_precision': precision,
        'training_audit': training_audit,
        'support_probe_output': probe_text, 'heldout_accessed': False,
        'gpu': torch.cuda.get_device_name(0), 'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(),
        'packages': {n: importlib.metadata.version(n) for n in ['torch','transformers','agenttune','trl','peft','datasets','bitsandbytes','accelerate']}}
    (output/'smoke.json').write_text(json.dumps(metadata, indent=2, default=str)+'\n')
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    commit = api.upload_folder(repo_id=repo, folder_path=output, path_in_repo=run_id,
                               ignore_patterns=['training/**'])
    metadata['artifact_commit'] = commit.oid
    encoded = base64.b64encode(json.dumps(metadata, default=str).encode()).decode()
    chunks = [encoded[i:i+12000] for i in range(0, len(encoded), 12000)]
    for index, chunk in enumerate(chunks, 1):
        print('SFT_TRAINING_RESULT_CHUNK='+json.dumps({'part': index, 'parts': len(chunks), 'data': chunk}), flush=True)
    print('SFT_SMOKE_RESULT='+json.dumps({key: value for key, value in metadata.items() if key != 'training_audit'}, default=str), flush=True)
    print(f'Private artifacts: https://huggingface.co/{repo}/tree/main/{run_id}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
