"""Balance the first native tool decision with the complete reviewed action."""
from __future__ import annotations

import copy
import math


def joint_weights(labels: list[int],weights: list[float],audit: dict,share: float=.75):
    positions=sorted({p for s in audit['spans'] if s['kind']=='tool_name' for p in s['label_positions']})
    active=audit['label_positions']
    if (not positions or not 0<share<1 or len(weights)!=len(labels)
        or any(weights[p]<=0 for p in active) or positions[0] not in active
        or any(weights[p]!=0 for p,v in enumerate(labels) if v==-100)):
        raise ValueError('invalid joint-objective inputs')
    normalizer=sum(weights);result=[(1-share)*w/normalizer for w in weights]
    result[positions[0]]+=share
    annotation=copy.deepcopy(audit)
    annotation.update(objective='joint_first_choice_completion',decision_position=positions[0],
        decision_share=share,normalizer=sum(result),base_target_weights=list(audit['target_weights']),
        target_weights=[result[p] for p in active])
    if not math.isclose(sum(result),1.,abs_tol=1e-12):raise ValueError('joint weights do not sum to one')
    return result,annotation


def loss_self_check(torch):
    from training.agenttune.weighted_loss import causal_weighted_loss
    logits=torch.zeros((1,5,7),device='cuda',requires_grad=True)
    labels=torch.tensor([[-100,-100,2,3,4]],device='cuda')
    vector,_=joint_weights(labels[0].tolist(),[0.,0.,1.,1.,1.],
        {'spans':[{'kind':'tool_name','label_positions':[2]}],
         'label_positions':[2,3,4],'target_weights':[1.,1.,1.]})
    weights=torch.tensor([vector],device='cuda')
    loss,_,_=causal_weighted_loss(logits,labels,weights)
    grad,=torch.autograd.grad(loss,logits)
    norms=grad[0].abs().sum(-1)
    if not (bool(norms[0]==0) and bool(norms[4]==0) and bool((norms[1:4]>0).all())
        and bool(norms[1]>norms[2]) and bool(torch.isfinite(loss))):
        raise RuntimeError('joint decision/completion gradient check failed')
    return {'prompt_logits_masked':True,'completion_logits_supervised':True,
        'decision_logits_emphasized':True,'model_parameters_updated':False}
