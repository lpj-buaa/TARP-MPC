from pathlib import Path
import hashlib
import numpy as np
import torch
from tarp_boundary.io import write
from tarp_sequence.response import ResponseModel
from .data import role,partition,public
from .core import LOGIT_TARGET


def encode(cases,checkpoint,scale,out,device):
    model=ResponseModel().to(device);model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model']);model.eval()
    result={};norms=[]
    with torch.no_grad():
        for c in cases:
            x=np.c_[np.repeat(c['state'][None],32,axis=0),c['action']]
            x=torch.as_tensor(x,device=device,dtype=torch.float32);h=model.encoder(x);p,l=model(x)
            hidden=h.cpu().numpy()
            result[c['key']+'__h']=hidden;result[c['key']+'__base']=np.c_[p.cpu().numpy(),l.cpu().numpy()/LOGIT_TARGET]
            if role(c)=='train':norms.extend(np.linalg.norm(hidden[c['initial']],axis=1).tolist())
    norm=max(float(np.mean(norms)),1e-6)
    errors=[]
    for c in cases:
        h=result.pop(c['key']+'__h');result[c['key']+'__features']=np.c_[h/norm,np.ones(32)]
        if role(c)=='train':
            ix=c['initial'];true=np.c_[c['response'][ix]/scale,2*(~c['recoverable'][ix].astype(bool)).astype(float)-1]
            errors.extend(result[c['key']+'__base'][ix]-true)
    noise=np.maximum(np.sqrt(np.mean(np.array(errors)**2,axis=0)),.05)
    result['noise_scale']=noise;result['feature_norm']=np.array(norm)
    np.savez_compressed(out,**result)


def load_bank(cases,path):
    with np.load(path) as f:
        encoded={c['key']:dict(features=f[c['key']+'__features'],base=f[c['key']+'__base'],public=public(c)) for c in cases}
        noise=f['noise_scale']
    refs={}
    for r in sorted({role(c) for c in cases}):
        items=[(c,i) for c in cases if role(c)==r for i in partition(c)['audit']]
        h=np.array([encoded[c['key']]['features'][i] for c,i in items]);base=np.array([encoded[c['key']]['base'][i] for c,i in items])
        # Coordinate selection is fixed and uses no outcomes.
        ordered=sorted(range(len(items)),key=lambda j:hashlib.sha256(f"probe|{items[j][0]['key']}|{items[j][1]}".encode()).hexdigest())
        refs[r]=dict(features=h,base=base,probe=ordered[:16],identities=[(c['key'],i) for c,i in items])
    return encoded,refs,noise


def reward_truth(cases,r):
    items=[(c,i) for c in cases if role(c)==r for i in partition(c)['audit']]
    return dict(response=np.array([c['response'][i] for c,i in items]),
        failure=np.array([~c['recoverable'][i].astype(bool) for c,i in items]),
        sources=[c['source_id'] for c,i in items])
