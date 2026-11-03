from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from tarp_boundary.io import write
from tarp_sequence.data import TRAIN_ROLES,TUNE_ROLES
from tarp_sequence.response import ResponseModel,continuous_loss
from tarp_interim.training import arrays,monitor
from .core import neighbor_weights,residual_tensor


def training_neighbors(cases, offsets, ids, rng):
    if any(c['role'] not in TRAIN_ROLES for c in cases):
        raise ValueError('Conditional supervision requires training contexts only')
    neighbors=[];weights=[]
    for ident in ids:
        context=int(np.searchsorted(offsets,ident,side='right')-1)
        c=cases[context];target=int(ident-offsets[context])
        initial=[i for i in c['initial'] if i!=target]
        available=[i for i in range(len(c['action'])) if i not in initial and i!=target]
        observed=initial+rng.permutation(available)[:int(rng.choice([0,4,8]))].tolist()
        # All complete pools have three initial branches; exclusion leaves at least two.
        ix,w=neighbor_weights(c['action'],observed,[target])
        row=ix[0].tolist();weight=w[0].tolist()
        while len(row)<3:row.append(row[-1]);weight.append(0.)
        neighbors.append(np.array(row)+offsets[context]);weights.append(weight)
    return np.array(neighbors),np.array(weights)


def fit_conditional(cases,out,seed=42,device='cpu',epochs=100):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(seed)
    rng=np.random.default_rng(seed);history_rng=np.random.default_rng(seed+9103)
    train=[c for c in cases if c['role'] in TRAIN_ROLES]
    tune=[c for c in cases if c['role'] in TUNE_ROLES]
    if not train or not tune:raise ValueError('Independent training and tuning required')
    training,refs,offsets=arrays(train,device);validation,vrefs,_=arrays(tune,device)
    X,Y,M,F,K=training;model=ResponseModel().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    pos=(F*K).sum(0);neg=((1-F)*K).sum(0);posweight=(neg/pos.clamp(min=1)).clamp(1,20)
    bce=nn.BCEWithLogitsLoss(reduction='none',pos_weight=posweight)
    best=np.inf;patience=0;history=[];steps=0;start=time.perf_counter()
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train();losses=[];conditional_losses=[]
        order=rng.permutation(len(X))
        for begin in range(0,len(X),128):
            ids=order[begin:begin+128]
            partner=np.array([rng.integers(offsets[np.searchsorted(offsets,i,side='right')-1],offsets[np.searchsorted(offsets,i,side='right')]) for i in ids])
            ix=torch.as_tensor(ids,device=device);jx=torch.as_tensor(partner,device=device)
            neighbors,weights=training_neighbors(train,offsets,ids,history_rng)
            nx=torch.as_tensor(neighbors,device=device);w=torch.as_tensor(weights,device=device,dtype=torch.float32)
            prediction,logits=model(X)
            pred,other,logit=prediction[ix],prediction[jx],logits[ix]
            corrected=residual_tensor(pred,prediction[nx],Y[nx],M[nx],w)
            conditional=continuous_loss(corrected,Y[ix],M[ix])
            base=continuous_loss(pred,Y[ix],M[ix])+.25*continuous_loss(pred-other,Y[ix]-Y[jx],M[ix]&M[jx])
            base=base+.2*(bce(logit,F[ix])*K[ix]).sum()/K[ix].sum().clamp(min=1)
            loss=base+.5*conditional
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            steps+=1;losses.append(float(loss.detach()));conditional_losses.append(float(conditional.detach()))
        tr,va=monitor(model,training,refs),monitor(model,validation,vrefs)
        history.append(dict(epoch=epoch+1,train=tr,validation=va,loss=float(np.mean(losses)),conditional_loss=float(np.mean(conditional_losses))))
        checkpoint=dict(model=model.state_dict(),seed=seed,epoch=epoch+1,conditional_weight=.5)
        torch.save(checkpoint,out/'last_model.pt')
        if va['selection_loss']<best-1e-7:
            best=va['selection_loss'];patience=0;torch.save(checkpoint,out/'model.pt')
        else:patience+=1
        if epoch==0 or (epoch+1)%20==0:
            print(f'conditional s{seed} epoch {epoch+1}: train={tr["selection_loss"]:.5f} val={va["selection_loss"]:.5f}',flush=True)
        if patience>=12:break
    checkpoint=torch.load(out/'model.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model']);model.eval();predictions={}
    with torch.no_grad():
        for c in cases:
            x=np.c_[np.repeat(c['state'][None],len(c['action']),axis=0),c['action']]
            p,l=model(torch.as_tensor(x,device=device,dtype=torch.float32))
            predictions[c['key']]=np.c_[p.cpu().numpy(),l.sigmoid().cpu().numpy()]
    np.savez_compressed(out/'predictions.npz',**predictions)
    write(out/'fit.json',dict(seed=seed,epochs=len(history),best_epoch=checkpoint['epoch'],history=history,
        parameters=sum(p.numel() for p in model.parameters()),optimizer_steps=steps,
        train_sources=sorted({c['source_id'] for c in train}),tune_sources=sorted({c['source_id'] for c in tune}),
        best_train=monitor(model,training,refs),best_validation=monitor(model,validation,vrefs),
        response_training_calls=1,elapsed_seconds=time.perf_counter()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
        selection='Unchanged static tuning criterion',conditional_weight=.5,new_simulations=0))
    return predictions
