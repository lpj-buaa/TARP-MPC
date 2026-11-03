from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from tarp_boundary.io import write
from tarp_sequence.response import ResponseModel,continuous_loss
from tarp_sequence.data import TRAIN_ROLES,TUNE_ROLES,GROUPS


def regression_metrics(prediction,target,mask,refs):
    p,y,m=np.asarray(prediction),np.asarray(target),np.asarray(mask,bool);refs=np.asarray(refs)
    def avg(x,valid):return float(np.mean(x[valid])) if valid.any() else None
    error=p-y;pm=m&m[refs]&(refs!=np.arange(len(refs)))[:,None]
    delta=(p-p[refs])-(y-y[refs]);true_delta=y-y[refs]
    significant=pm&(np.abs(true_delta)>.01)
    result=dict(known=int(m.sum()),mae=avg(np.abs(error),m),mse=avg(error**2,m),zero_mae=avg(np.abs(y),m),
                delta_known=int(pm.sum()),delta_mae=avg(np.abs(delta),pm),delta_zero_mae=avg(np.abs(true_delta),pm),
                direction_known=int(significant.sum()),direction_accuracy=avg((np.sign(p-p[refs])==np.sign(true_delta)).astype(float),significant))
    result['groups']={g:dict(mae=avg(np.abs(error[:,lo:hi]),m[:,lo:hi]),mse=avg(error[:,lo:hi]**2,m[:,lo:hi]),
                            known=int(m[:,lo:hi].sum())) for g,(lo,hi) in GROUPS.items()}
    return result


def arrays(cases,device):
    x=np.concatenate([np.c_[np.repeat(c['state'][None],len(c['action']),axis=0),c['action']] for c in cases])
    offsets=np.cumsum([0]+[len(c['action']) for c in cases])
    refs=np.concatenate([np.full(len(c['action']),offsets[j]+c['families'].index('reference')) for j,c in enumerate(cases)])
    a=[torch.tensor(v,device=device,dtype=d) for v,d in [(x,torch.float32),
       (np.concatenate([c['response'] for c in cases]),torch.float32),(np.concatenate([c['response_mask'] for c in cases]),torch.bool),
       (np.concatenate([c['gaps']>1e-6 for c in cases]),torch.float32),(np.concatenate([c['recovery_mask'] for c in cases]),torch.bool)]]
    return a,refs,offsets


def monitor(model,a,refs):
    model.eval();x,y,m,f,k=a
    with torch.no_grad():
        p,l=model(x)
        criterion=continuous_loss(p,y,m)+.2*(nn.functional.binary_cross_entropy_with_logits(l,f,reduction='none')*k).sum()/k.sum().clamp(min=1)
        result=regression_metrics(p.cpu().numpy(),y.cpu().numpy(),m.cpu().numpy(),refs)
        result['selection_loss']=float(criterion)
    return result


def fit_response(cases,out,seed=42,device='cpu',epochs=100):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    start=time.perf_counter()
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    train=[c for c in cases if c['role'] in TRAIN_ROLES];tune=[c for c in cases if c['role'] in TUNE_ROLES]
    if not train or not tune:raise ValueError('Independent training/tuning contexts required')
    training,trainrefs,offsets=arrays(train,device);validation,valrefs,_=arrays(tune,device)
    X,Y,M,F,K=training;model=ResponseModel().to(device);opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    positives=(F*K).sum(0);negatives=((1-F)*K).sum(0);posweight=(negatives/positives.clamp(min=1)).clamp(1,20)
    bce=nn.BCEWithLogitsLoss(reduction='none',pos_weight=posweight)
    best=np.inf;patience=0;history=[];steps=0
    for epoch in range(epochs):
        model.train();losses=[];order=rng.permutation(len(X))
        for begin in range(0,len(X),128):
            ids=order[begin:begin+128]
            partner=np.array([rng.integers(offsets[np.searchsorted(offsets,i,side='right')-1],offsets[np.searchsorted(offsets,i,side='right')]) for i in ids])
            ix=torch.tensor(ids,device=device);jx=torch.tensor(partner,device=device)
            pred,logit=model(X[ix]);other,_=model(X[jx])
            loss=continuous_loss(pred,Y[ix],M[ix])+.25*continuous_loss(pred-other,Y[ix]-Y[jx],M[ix]&M[jx])+.2*(bce(logit,F[ix])*K[ix]).sum()/K[ix].sum().clamp(min=1)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1;losses.append(float(loss.detach()))
        train_metrics=monitor(model,training,trainrefs);val_metrics=monitor(model,validation,valrefs)
        history.append(dict(epoch=epoch+1,train=train_metrics,validation=val_metrics,optimization_loss=float(np.mean(losses))))
        val=val_metrics['selection_loss']
        checkpoint=dict(model=model.state_dict(),seed=seed,epoch=epoch+1,scope='interim monitored original architecture; not formal confirm')
        torch.save(checkpoint,out/'last_model.pt')
        if val<best-1e-7:
            best=val;patience=0;torch.save(checkpoint,out/'model.pt')
        else:patience+=1
        if (epoch+1)%20==0 or epoch==0:print(f'response s{seed} epoch {epoch+1}: train={train_metrics["selection_loss"]:.5f} val={val:.5f}',flush=True)
        if patience>=12:break
    checkpoint=torch.load(out/'model.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['model']);model.eval();predictions={}
    with torch.no_grad():
        for c in cases:
            x=np.c_[np.repeat(c['state'][None],len(c['action']),axis=0),c['action']]
            p,l=model(torch.tensor(x,device=device,dtype=torch.float32));predictions[c['key']]=np.c_[p.cpu().numpy(),l.sigmoid().cpu().numpy()]
    np.savez_compressed(out/'predictions.npz',**predictions)
    result=dict(seed=seed,epochs=len(history),best_epoch=checkpoint['epoch'],history=history,
        best_train=monitor(model,training,trainrefs),best_validation=monitor(model,validation,valrefs),
        train_sources=sorted({c['source_id'] for c in train}),tune_sources=sorted({c['source_id'] for c in tune}),
        parameters=sum(p.numel() for p in model.parameters()),optimizer_steps=steps,device=device,
        elapsed_seconds=time.perf_counter()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
        response_training_calls=1,old_model_training_calls=0,positive_weights=posweight.cpu().tolist(),
        monitoring_random_draws=0,selection='Original tuning roles only; same unweighted validation criterion as V4')
    write(out/'fit.json',result)
    return predictions
