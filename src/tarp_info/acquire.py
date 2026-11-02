"""Matched contextual actor-critic acquisition in a shared finite action pool."""
from collections import defaultdict
import numpy as np
from scipy.stats import qmc


def novelty(z,key,known,exclude=None):
    values=[v for id,v in known.get(key,{}).items() if id!=exclude and v is not None]
    return float(np.min(np.linalg.norm(np.asarray(values)-z,axis=1))) if values else 0.


def sobol_targets(public,budget,seed):
    keys=sorted(set(p['key'] for p in public));count=int(np.ceil(budget/len(keys)))
    result={}
    for i,key in enumerate(keys):
        actions=np.asarray([p['action'] for p in public if p['key']==key])
        low=actions.min(0);high=actions.max(0)
        points=qmc.Sobol(actions.shape[1],scramble=True,seed=seed+i*1009).random_base2(int(np.ceil(np.log2(max(count,1)))))[:count]
        result[key]=low+(high-low)*points
    return result


def run_arm(public,initial,oracle,method,budget,seed=42,warm=100,updates=10):
    if method not in ['random','sobol','error_td3','space_td3']:raise ValueError(method)
    if budget>len(public):raise ValueError('Insufficient common candidates')
    rng=np.random.default_rng(seed); keys=sorted(set(p['key'] for p in public));pool=defaultdict(list)
    for p in public:pool[p['key']].append(p)
    known=defaultdict(dict); data=[]
    for x in initial:
        z=None if x['profile'] is None else np.asarray(x['profile'])
        known[x['key']][x['id']]=z;data.append(dict(x,profile=z))
    agent=None;torch=None
    if method.endswith('td3'):
        import torch
        from tarp37.acquisition import AcquisitionTD3
        torch.set_num_threads(1);agent=AcquisitionTD3(len(public[0]['state']),'cpu',seed)
    def reward(x):
        if method=='error_td3':return min(x['error'],5.)
        return novelty(x['profile'],x['key'],known,exclude=x['id']) if x['profile'] is not None else -.25
    def refresh():
        # Fitted non-stationary contextual acquisition, gamma=0; no stationary
        # MDP guarantee claimed. All labels here have already been revealed.
        agent.replay=[]
        for x in data:agent.add(x['state'],x['action'],reward(x))
    if agent:
        refresh()
        for _ in range(warm):agent.update(32)
    targets=sobol_targets(public,budget,seed) if method=='sobol' else None
    visits=defaultdict(int)
    queries=[]
    for i in range(budget):
        key=keys[i%len(keys)]
        if not pool[key]:key=next(k for k in keys if pool[k])
        candidates=pool[key]
        if method=='random':j=int(rng.integers(len(candidates)))
        elif method=='sobol':j=int(np.argmin(np.linalg.norm(np.asarray([c['action'] for c in candidates])-targets[key][visits[key]],axis=1)))
        elif rng.random()<.25:j=int(rng.integers(len(candidates)))
        else:
            sa=torch.tensor([c['state']+c['action'] for c in candidates],dtype=torch.float32)
            with torch.no_grad():scores=torch.minimum(agent.q1(sa),agent.q2(sa)).numpy().ravel()
            j=int(np.argmax(scores))
        selected=candidates.pop(j)
        visits[key]+=1
        a=oracle.query(selected['id']);z=None if a['profile'] is None else np.asarray(a['profile'])
        insertion=novelty(z,key,known) if z is not None else None
        x=dict(selected,**a);x['profile']=z
        known[key][x['id']]=z;data.append(x)
        queries.append(dict(id=x['id'],key=key,source_id=x.get('source_id',key),profile=a['profile'],error=a['error'],
            event=a['event'],known=a['known'],late_service=a['late_service'],induced_joint=a.get('induced_joint',False),
            insertion_novelty=insertion))
        if agent:
            refresh()
            for _ in range(updates):agent.update(32)
    return dict(method=method,seed=seed,budget=budget,queries=queries,oracle_calls=list(oracle.calls),
        acquisition_training_calls=int(agent is not None),risk_forecaster_training_calls=0,
        selection='25% exploration, otherwise twin-critic finite-pool ranking; actor updated but not used for proposal',
        gamma=0,initial_rows=len(initial))
