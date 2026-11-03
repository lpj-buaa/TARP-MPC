from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from tarp_boundary.io import write,read
from tarp_sequence.response import ResponseModel,continuous_loss
from .data import role,partition
from .core import task_errors


def arrays(rows,scale,device):
    if not rows:raise ValueError('No exposed training rows')
    for c,i in rows:
        if not c['response_mask'][i].all() or not c['recovery_mask'][i].all():raise ValueError('Unknown supervised row')
    x=np.array([np.r_[c['state'],c['action'][i]] for c,i in rows])
    y=np.array([c['response'][i]/scale for c,i in rows]);f=np.array([~c['recoverable'][i].astype(bool) for c,i in rows],float)
    return [torch.as_tensor(v,device=device,dtype=torch.float32) for v in [x,y,f]]


def predict(model,rows,scale,device):
    x=np.array([np.r_[c['state'],c['action'][i]] for c,i in rows])
    with torch.no_grad():p,l=model(torch.as_tensor(x,device=device,dtype=torch.float32))
    return np.c_[p.cpu().numpy()*scale,l.cpu().numpy()]


def fit_network(cases,rows,out,scale,seed,device,initial=None,epochs=100):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(seed)
    if any(role(c)!='train' or i in partition(c)['audit'] for c,i in rows):raise ValueError('Non-exposed target entered learner fit')
    validation=[(c,i) for c in cases if role(c)=='tune' for i in (partition(c)['initial'] if initial is None else partition(c)['audit'])]
    if not validation:raise ValueError('No independent tuning rows')
    model=ResponseModel().to(device)
    if initial:model.load_state_dict(torch.load(initial,map_location=device,weights_only=False)['model'])
    X,Y,F=arrays(rows,scale,device);mask=torch.ones_like(Y,dtype=torch.bool)
    pos=F.sum(0);weight=((1-F).sum(0)/pos.clamp(min=1)).clamp(1,20)
    bce=nn.BCEWithLogitsLoss(pos_weight=weight)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4);rng=np.random.default_rng(seed)
    groups={}
    for n,(c,i) in enumerate(rows):groups.setdefault(c['key'],[]).append(n)
    best=np.inf;patience=0;history=[];steps=0;start=time.perf_counter()
    valy=np.array([c['response'][i] for c,i in validation]);valf=np.array([~c['recoverable'][i].astype(bool) for c,i in validation])
    for epoch in range(epochs):
        model.train();losses=[]
        for ids in np.array_split(rng.permutation(len(rows)),max(1,int(np.ceil(len(rows)/128)))):
            ix=torch.as_tensor(ids,device=device);partners=[rng.choice(groups[rows[int(i)][0]['key']]) for i in ids]
            jx=torch.as_tensor(partners,device=device);p,l=model(X[ix]);other,_=model(X[jx])
            loss=continuous_loss(p,Y[ix],mask[ix])+.25*continuous_loss(p-other,Y[ix]-Y[jx],mask[ix])+.2*bce(l,F[ix])
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1;losses.append(float(loss.detach()))
        model.eval();raw=predict(model,validation,scale,device);error=task_errors(raw,valy,valf,scale)
        by={s:float(np.mean([e for (c,i),e in zip(validation,error) if c['source_id']==s])) for s in {c['source_id'] for c,i in validation}}
        value=float(np.mean(list(by.values())))
        history.append(dict(epoch=epoch+1,optimization_loss=float(np.mean(losses)),validation_error=value,validation_sources=by))
        if value<best-1e-6:
            best=value;patience=0;torch.save(dict(model=model.state_dict(),seed=seed,epoch=epoch+1,scaled_continuous_outputs=True),out/'model.pt')
        else:patience+=1
        if patience>=12:break
    write(out/'fit.json',dict(seed=seed,epochs=len(history),best_epoch=read_best_epoch(out,device),best_validation=best,
        history=history,parameters=sum(p.numel() for p in model.parameters()),optimizer_steps=steps,
        rows=[dict(key=c['key'],source_id=c['source_id'],index=i,id=c['ids'][i]) for c,i in rows],
        validation_role='tune',validation_targets='initial' if initial is None else 'audit',
        seconds=time.perf_counter()-start,network_training_calls=1,new_simulations=0))


def read_best_epoch(out,device):
    return int(torch.load(Path(out)/'model.pt',map_location=device,weights_only=False)['epoch'])
