"""Deterministic completion-token weights; approved text is never rewritten."""
from __future__ import annotations

import difflib
import json
import re


def initialize_loss_scale(accelerator, initial_scale: float) -> dict:
    """Reduce AMP startup scale before optimizer creation; leave gradients intact."""
    import math
    if not math.isfinite(initial_scale) or initial_scale <= 0:
        raise ValueError('invalid AMP initial scale')
    scaler = accelerator.scaler
    if scaler is None or not scaler.is_enabled():
        raise ValueError('weighted T4 experiment requires enabled FP16 AMP')
    before = scaler.state_dict()
    after = {**before, 'scale':float(initial_scale)}
    scaler.load_state_dict(after)
    return {'before':before, 'after':scaler.state_dict()}


def old_file_content(row: dict, path: str) -> str:
    messages = row['prompt']
    for index in range(len(messages)-2, -1, -1):
        if messages[index]['role'] != 'assistant' or messages[index+1]['role'] != 'user':
            continue
        try:
            action = json.loads(messages[index]['content'])[0]
            if action['tool_name'] != 'read_file' or action['parameters']['path'] != path:
                continue
            result = json.loads(messages[index+1]['content'].split('\n',1)[1])
            content = result['stdout']
            if (result['success'] is not True or not isinstance(content,str)
                or result.get('stdout_truncated',False) or action['parameters'].get('start_line',1)!=1):
                continue
            count=len(content.splitlines())
            total=result.get('metadata',{}).get('line_count')
            end=action['parameters'].get('end_line')
            # Runtime drops metadata. A nontruncated read from line 1 that
            # returns fewer lines than requested has reached EOF. A slice
            # exactly as long as the request is not evidence of a full file.
            complete=(total==count if total is not None else
                end is None or (isinstance(end,int) and not isinstance(end,bool) and count<end))
            if complete:return content
        except (ValueError, KeyError, TypeError, IndexError):
            continue
    raise ValueError('changed-code weights require an earlier complete file observation')


def changed_ranges(old: str, new: str) -> list[tuple[int,int]]:
    spans = []
    for tag, _a, _b, left, right in difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        if left == right:  # deleted characters have no target; mark adjacent boundary
            left, right = max(0,left-1), min(len(new),right+1)
        if left < right:
            spans.append((left,right))
    return spans


def action_spans(row: dict, scheme: dict) -> list[dict]:
    gold = row['completion'][0]['content']
    action = json.loads(gold)[0]
    tool = re.search(r'"tool_name"\s*:\s*"([^"]+)"',gold)
    closure = re.search(r'[}\]\s]+$',gold)
    if tool is None or closure is None:
        raise ValueError('cannot locate tool-name or closure span')
    spans = [{'kind':'tool_name','start':tool.start(1),'end':tool.end(1),'weight':scheme['tool_name']},
             {'kind':'json_closure','start':closure.start(),'end':closure.end(),'weight':scheme['json_closure']}]
    if action['tool_name'] == 'write_file':
        new = action['parameters']['content']
        old = old_file_content(row,action['parameters']['path'])
        match = re.search(r'"content"\s*:\s*',gold)
        if match is None:
            raise ValueError('missing write content')
        decoded, length = json.JSONDecoder().raw_decode(gold[match.end():])
        if decoded != new:
            raise ValueError('ambiguous write content span')
        encoded = ''.join(json.dumps(char,ensure_ascii=False)[1:-1] for char in new)
        if gold[match.end():match.end()+length] != '"'+encoded+'"':
            raise ValueError('unsupported string escaping in approved action')
        offsets = [0]
        for char in new:
            offsets.append(offsets[-1]+len(json.dumps(char,ensure_ascii=False)[1:-1]))
        for left,right in changed_ranges(old,new):
            spans.append({'kind':'changed_code','start':match.end()+1+offsets[left],
                          'end':match.end()+1+offsets[right],'weight':scheme['changed_code']})
        if not any(s['kind']=='changed_code' for s in spans):
            raise ValueError('write reference has no changed code')
    return spans


def align_weights(labels: list[int], offsets: list, spans: list[dict], gold_start: int, base: float=1.0) -> tuple[list[float],dict]:
    if len(labels) != len(offsets):
        raise ValueError('token offsets and labels differ in length')
    weights = [0.0 if value == -100 else float(base) for value in labels]
    mapped = []
    for span in spans:
        positions = [i for i,(a,b) in enumerate(offsets) if labels[i] != -100
                     and a < gold_start+span['end'] and b > gold_start+span['start']]
        if not positions:
            raise ValueError('critical span did not map to supervised tokens')
        for index in positions:
            weights[index] = max(weights[index],float(span['weight']))
        mapped.append({**span,'label_positions':positions})
    return weights, {'spans':mapped,'normalizer':sum(weights),'target_tokens':sum(v!=-100 for v in labels)}


def token_weights(tokenizer,row: dict,ids: list[int],labels: list[int],scheme: dict) -> tuple[list[float],dict]:
    rendered = tokenizer.apply_chat_template(row['prompt']+row['completion'],tokenize=False)
    encoded = tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids'] != ids:
        raise ValueError('rendered offsets do not match trainer input IDs')
    gold = row['completion'][0]['content']
    start = rendered.rfind(gold)
    if start < 0:
        raise ValueError('gold action missing from rendered training text')
    prompt_ids = tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,return_dict=True)['input_ids']
    active = [i for i,v in enumerate(labels) if v!=-100]
    if not active or ids[:active[0]] != prompt_ids or active != list(range(active[0],len(labels))):
        raise ValueError('completion labels/prompt boundary changed')
    if labels[-1] != tokenizer.eos_token_id:
        raise ValueError('EOS must remain supervised')
    weights,audit = align_weights(labels,encoded['offset_mapping'],action_spans(row,scheme),start,scheme['base'])
    if weights[-1] != scheme['base']:
        raise ValueError('EOS weight unexpectedly changed')
    audit.update(source_example_id=row['source_example_id'],source_task_id=row['source_task_id'],
        expected_tool=json.loads(gold)[0]['tool_name'],label_positions=active,
        target_token_ids=[labels[i] for i in active],target_weights=[weights[i] for i in active])
    return weights,audit


def causal_weighted_loss(logits, labels, weights):
    import torch
    import torch.nn.functional as F
    if labels.shape != weights.shape or logits.shape[:2] != labels.shape:
        raise ValueError('loss tensors have incompatible shapes')
    if not torch.isfinite(weights).all() or (weights<0).any() or (weights[labels==-100]!=0).any():
        raise ValueError('invalid or unmasked context weights')
    shifted = labels[:,1:]
    active = shifted!=-100
    coefficients = weights[:,1:][active].float()
    if not active.any() or (coefficients<=0).any():
        raise ValueError('missing positive completion weights')
    scores = logits[:,:-1,:][active].float()
    losses = F.cross_entropy(scores,shifted[active],reduction='none')
    return (losses*coefficients).sum()/coefficients.sum(), losses.mean(), float(coefficients.sum())


def weighted_trainer_class(parent, *, tool_choice_only: bool=False):
    class DecisionWeightedTrainer(parent):
        def _set_signature_columns_if_needed(self):
            super()._set_signature_columns_if_needed()
            if 'loss_weights' not in self._signature_columns:
                self._signature_columns.append('loss_weights')

        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            if self.args.gradient_accumulation_steps!=1 or self.accelerator.num_processes!=1:
                raise ValueError('weighted objective is fixed to batch-one single-GPU updates')
            inputs = dict(inputs)
            labels,weights = inputs.pop('labels'),inputs.pop('loss_weights')
            inputs.pop('_prediction_loss_only',None)
            outputs = model(**inputs,use_cache=False)
            if tool_choice_only:
                from training.agenttune.tool_choice_objective import causal_tool_choice_loss
                loss,unweighted,normalizer = causal_tool_choice_loss(outputs.logits,labels,weights)
            else:
                loss,unweighted,normalizer = causal_weighted_loss(outputs.logits,labels,weights)
            self._last_weighted_metrics = {'unweighted_completion_loss':float(unweighted.detach()),'completion_weight_sum':normalizer}
            return (loss,outputs) if return_outputs else loss

        def log(self,logs,*args,**kwargs):
            if 'loss' in logs:
                logs = {**logs,**getattr(self,'_last_weighted_metrics',{})}
            return super().log(logs,*args,**kwargs)
    return DecisionWeightedTrainer


def weighted_collator(original):
    def collate(features):
        import torch
        vectors = [row['loss_weights'] for row in features]
        batch = original([{k:v for k,v in row.items() if k!='loss_weights'} for row in features])
        result = torch.zeros_like(batch['labels'],dtype=torch.float32)
        # The fixed experiment uses right padding and rejects truncation upstream.
        for index,vector in enumerate(vectors):
            if len(vector)>result.shape[1]:
                raise ValueError('loss weights would be truncated')
            result[index,:len(vector)] = torch.tensor(vector,dtype=torch.float32)
        result[batch['labels']==-100] = 0
        batch['loss_weights'] = result
        return batch
    return collate
