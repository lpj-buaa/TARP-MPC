from pathlib import Path
import numpy as np
import torch
from torch import nn
from tarp_boundary.io import read,write,sha
from .data import GROUPS,TRAIN_ROLES,TUNE_ROLES


def continuous_loss(prediction,target,mask):
    losses=[]
    for lo,hi in GROUPS.values():
        m=mask[:,lo:hi];error=torch.where(m,prediction[:,lo:hi]-target[:,lo:hi],0.)
        losses.append(error.square().sum()/m.sum().clamp(min=1))
    return torch.stack(losses).mean()


class ResponseModel(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(874,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.continuous=nn.Linear(128,60);self.failure=nn.Linear(128,3)
    def forward(self,x):
        h=self.encoder(x);return self.continuous(h),self.failure(h)


def fit(cases,out,seed=42,device='cpu',smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    train=[c for c in cases if c['role'] in TRAIN_ROLES];tune=[c for c in cases if c['role'] in TUNE_ROLES]
    if not tune:
        if not smoke:raise ValueError('No independent tuning contexts')
        tune=train
    def arrays(rows):
        x=np.concatenate([np.c_[np.repeat(c['state'][None],len(c['action']),axis=0),c['action']] for c in rows])
        return [torch.tensor(a,device=device,dtype=d) for a,d in [(x,torch.float32),
            (np.concatenate([c['response'] for c in rows]),torch.float32),(np.concatenate([c['response_mask'] for c in rows]),torch.bool),
            (np.concatenate([c['gaps']>1e-6 for c in rows]),torch.float32),(np.concatenate([c['recovery_mask'] for c in rows]),torch.bool)]]
    X,Y,M,F,K=arrays(train);validation=arrays(tune);model=ResponseModel().to(device);opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    positives=(F*K).sum(0);negatives=((1-F)*K).sum(0);posweight=(negatives/positives.clamp(min=1)).clamp(1,20)
    bce=nn.BCEWithLogitsLoss(reduction='none',pos_weight=posweight)
    offsets=np.cumsum([0]+[len(c['action']) for c in train])
    best=np.inf;patience=0;history=[]
    for epoch in range(3 if smoke else 100):
        model.train()
        order=rng.permutation(len(X))
        for begin in range(0,len(X),128):
            ids=order[begin:begin+128];partner=np.array([rng.integers(offsets[np.searchsorted(offsets,i,side='right')-1],offsets[np.searchsorted(offsets,i,side='right')]) for i in ids])
            ix=torch.tensor(ids,device=device);jx=torch.tensor(partner,device=device)
            pred,logit=model(X[ix]);other,_=model(X[jx])
            continuous=continuous_loss(pred,Y[ix],M[ix])
            pair=continuous_loss(pred-other,Y[ix]-Y[jx],M[ix]&M[jx])
            boundary=(bce(logit,F[ix])*K[ix]).sum()/K[ix].sum().clamp(min=1)
            loss=continuous+.25*pair+.2*boundary
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval()
        with torch.no_grad():
            vx,vy,vm,vf,vk=validation;p,l=model(vx)
            val=float(continuous_loss(p,vy,vm)+.2*(nn.functional.binary_cross_entropy_with_logits(l,vf,reduction='none')*vk).sum()/vk.sum().clamp(min=1))
        history.append(dict(epoch=epoch+1,validation=val))
        if val<best-1e-7:
            best=val;patience=0;torch.save(dict(model=model.state_dict(),seed=seed,epoch=epoch+1,scope='stable physical outcomes, not warning probabilities'),out/'model.pt')
        else:patience+=1
        if patience>=12:break
    checkpoint=torch.load(out/'model.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['model']);model.eval();predictions={}
    with torch.no_grad():
        for c in cases:
            x=np.c_[np.repeat(c['state'][None],len(c['action']),axis=0),c['action']]
            p,l=model(torch.tensor(x,device=device,dtype=torch.float32));predictions[c['key']]=np.c_[p.cpu().numpy(),l.sigmoid().cpu().numpy()]
    np.savez_compressed(out/'predictions.npz',**predictions)
    write(out/'fit.json',dict(seed=seed,epochs=len(history),best_epoch=checkpoint['epoch'],history=history,
        train_sources=sorted({c['source_id'] for c in train}),tune_sources=sorted({c['source_id'] for c in tune}),
        failure_counts=positives.cpu().tolist(),label_counts=K.sum(0).cpu().tolist(),positive_weights=posweight.cpu().tolist(),
        response_training_calls=1,old_model_training_calls=0,smoke_resubstitution=bool(smoke)))
    return predictions


def predicted_descriptor(prediction):
    p=np.asarray(prediction);bounds=p[:,:24].reshape(-1,3,4,2);width=bounds[:,:,:,1]-bounds[:,:,:,0]
    recover=p[:,60:63]<.5
    # The energy/rate bound is necessary only: it can certify a deficit, never full network recoverability.
    recover &= (p[:,24:36].reshape(-1,3,4)<=1+1e-6).all(-1)
    gap=np.maximum(p[:,36:39],0.)
    signs=np.where(np.abs(width)>.01,np.sign(width),0);turn=(signs>0).any(1)&(signs<0).any(1)
    return np.c_[width.reshape(-1,12)/np.sqrt(12),gap/np.sqrt(3),recover/np.sqrt(3),turn/2]/2
