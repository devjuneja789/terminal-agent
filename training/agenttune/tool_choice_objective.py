"""Isolate native tool_name learning without rewriting any reviewed action."""
from __future__ import annotations

import copy


def choice_weights(labels: list[int], audit: dict) -> tuple[list[float],dict]:
    positions=sorted({p for s in audit['spans'] if s['kind']=='tool_name' for p in s['label_positions']})
    if not positions or any(p<=0 or p>=len(labels) or labels[p] in (-100,6,261001) for p in positions):
        raise ValueError('invalid tool-choice supervision positions')
    weights=[0.0]*len(labels)
    for p in positions:weights[p]=1.0
    result=copy.deepcopy(audit)
    result.update(objective='tool_choice_only',selected_tool_positions=positions,normalizer=float(len(positions)),
        target_weights=[weights[p] for p in audit['label_positions']],
        zero_weight_completion_tokens=sum(labels[p]!=-100 and weights[p]==0 for p in range(len(labels))))
    return weights,result


def causal_tool_choice_loss(logits,labels,weights):
    import torch
    import torch.nn.functional as F
    if labels.shape!=weights.shape or logits.shape[:2]!=labels.shape:
        raise ValueError('choice loss tensor shapes differ')
    if (not torch.isfinite(weights).all() or (weights<0).any()
        or (weights[labels==-100]!=0).any() or not ((weights==0)|(weights==1)).all()):
        raise ValueError('choice loss requires binary weights and masked context')
    shifted=labels[:,1:];all_active=shifted!=-100
    selected=all_active & (weights[:,1:]>0)
    if not selected.any():raise ValueError('no selected tool-name labels')
    # Compute gradients only for the selected causal predictions. The full
    # completion NLL below is detached diagnostic logging, never optimization.
    choice=F.cross_entropy(logits[:,:-1,:][selected].float(),shifted[selected])
    with torch.no_grad():
        full=F.cross_entropy(logits[:,:-1,:][all_active].detach().float(),shifted[all_active])
    return choice,full,float(selected.sum())


def loss_self_check(torch) -> dict:
    logits=torch.zeros((1,5,7),device='cuda',requires_grad=True)
    labels=torch.tensor([[-100,-100,2,3,4]],device='cuda')
    weights=torch.tensor([[0.,0.,1.,0.,0.]],device='cuda')
    loss,_,_=causal_tool_choice_loss(logits,labels,weights)
    grad,=torch.autograd.grad(loss,logits)
    changed=logits.detach().clone();changed[0,2:,0]=100
    altered,_,_=causal_tool_choice_loss(changed,labels,weights)
    if (not torch.equal(loss.detach(),altered) or not bool(grad[0,1].abs().sum()>0)
        or bool(grad[0,[0,2,3,4]].abs().sum()!=0)):
        raise RuntimeError('tool-only objective gradient isolation failed')
    return {'nonchoice_logit_invariance':True,'only_selected_causal_position_has_gradient':True,
        'dummy_logits_only':True,'model_parameters_updated':False}


def gradient_preflight(trainer,torch,scale: float) -> dict:
    """Check every real batch at a fixed AMP scale, without optimizer updates."""
    if trainer.optimizer is not None:raise RuntimeError('gradient preflight must precede optimizer creation')
    model=trainer.model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train();records=[]
    for index in range(len(trainer.train_dataset)):
        batch=trainer.data_collator([trainer.train_dataset[index]])
        batch={k:v.to('cuda') for k,v in batch.items()}
        model.zero_grad(set_to_none=True)
        with trainer.accelerator.autocast():
            loss=trainer.compute_loss(model,batch)
        (loss*scale).backward()
        gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        finite=bool(torch.isfinite(loss)) and bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
        norm=float(torch.sqrt(sum(g.detach().float().square().sum() for g in gradients))/scale) if finite else None
        if not finite or not norm or norm<=0:raise RuntimeError(f'nonfinite/zero gradient at fixed scale {scale}, batch {index}; no updates applied')
        records.append({'dataset_index':index,'loss':float(loss.detach()),'unscaled_grad_norm':norm,'all_gradients_finite':True})
        model.zero_grad(set_to_none=True)
        del loss,batch,gradients
    if trainer.optimizer is not None or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('preflight left optimizer state or gradients')
    return {'scale':scale,'batch_count':len(records),'batches':records,'optimizer_updates':0,
        'all_gradients_finite':True,'gradients_cleared':True}
