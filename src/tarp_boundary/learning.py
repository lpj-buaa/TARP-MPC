"""Same terminal twin critics; new rewards consume revealed evidence only."""
import numpy as np
from collections import defaultdict


def knowledge_features(x,observed):
    known=[r for r in observed if r['key']==x['key'] and r['id']!=x['id']]
    nearest=min((np.linalg.norm(np.array(x['action'])-r['action'])/np.sqrt(51) for r in known),default=1.)
    family=[r for r in known if r['family'] in [x['family'],'reference'] and r.get('recoverable') is not None]
    lower=max((r for r in family if r['dose']<=x['dose']),key=lambda r:r['dose'],default=None)
    upper=min((r for r in family if r['dose']>=x['dose']),key=lambda r:r['dose'],default=None)
    gaps=[x['dose']-lower['dose'] if lower else 1.,upper['dose']-x['dose'] if upper else 1.]
    recover=(list(lower['recoverable']) if lower else [.5]*3)+(list(upper['recoverable']) if upper else [.5]*3)
    turns=(list(lower['turnover']) if lower else [.5]*4)+(list(upper['turnover']) if upper else [.5]*4)
    return np.asarray([min(nearest,1.),*gaps,*recover,*turns,min(len(known)/27,1.)],float).tolist()


def information_reward(x,observed):
    if x.get('descriptor') is None:return -.25
    others=[y for y in observed if y['id']!=x['id'] and y['key']==x['key'] and y.get('descriptor') is not None]
    if not others:return 0.
    novelty=min(1.,min(np.linalg.norm(np.array(x['descriptor'])-y['descriptor']) for y in others)/.1)
    neighbors=[y for y in others if (y['family']==x['family'] or 'reference' in [x['family'],y['family']]) and abs(y['dose']-x['dose'])<=1/13+1e-9]
    boundary=max([float(np.mean(np.array(x['recoverable'])!=y['recoverable'])) for y in neighbors],default=0.)
    turn=max([float(np.mean(np.array(x['turnover'])!=y['turnover'])) for y in neighbors],default=0.)
    return float((novelty+boundary+turn)/3.)


def bracket_proposal(candidates,observed):
    proposals=[]
    for key,family in sorted({(p['key'],p['family']) for p in candidates}):
        rows=sorted([x for x in observed if x['key']==key and x['family'] in [family,'reference'] and x.get('recoverable') is not None],key=lambda x:x['dose'])
        for left,right in zip(rows,rows[1:]):
            if left['recoverable']==right['recoverable']:continue
            inside=[p for p in candidates if p['key']==key and p['family']==family and left['dose']<p['dose']<right['dose']]
            if not inside:continue
            middle=(left['dose']+right['dose'])/2
            choice=min(inside,key=lambda p:(abs(p['dose']-middle),p['id']))
            proposals.append((right['dose']-left['dose'],choice))
    return max(proposals,key=lambda p:(p[0],p[1]['id']))[1] if proposals else None


def acquire(public,initial,oracle,method,budget=128,seed=42,checkpoint_path=None):
    from tarp_info.acquire import sobol_targets
    methods=['random','sobol','error_td3','boundary_reward_only','bracket_random','boundary_td3']
    if method not in methods:raise ValueError(method)
    rng=np.random.default_rng(seed);keys=sorted({p['key'] for p in public});pool=defaultdict(list);visits=defaultdict(int)
    for p in public:pool[p['key']].append(p)
    observed=[dict(x) for x in initial];agent=None;torch=None
    if method in ['error_td3','boundary_reward_only','boundary_td3']:
        import torch
        from tarp37.acquisition import AcquisitionTD3
        torch.set_num_threads(1);agent=AcquisitionTD3(len(public[0]['state'])+18,'cpu',seed)
    def state(x):return x['state']+knowledge_features(x,observed)
    def refresh():
        agent.replay=[]
        for x in observed:
            reward=min(x['error'],5.) if method=='error_td3' else information_reward(x,observed)
            agent.add(state(x),x['action'],reward)
    if agent:
        refresh()
        for _ in range(100):agent.update(32)
    targets=sobol_targets(public,budget,seed) if method=='sobol' else None
    queries=[]
    for i in range(budget):
        key=keys[i%len(keys)];candidates=pool[key];proposal=None;path='rank'
        if method in ['boundary_td3','bracket_random'] and visits[key]%2==1:
            proposal=bracket_proposal(candidates,observed);path='observed-bracket' if proposal else 'rank'
        if proposal is None:
            if method in ['random','bracket_random']:proposal=candidates[int(rng.integers(len(candidates)))];path='random'
            elif method=='sobol':proposal=candidates[int(np.argmin(np.linalg.norm(np.array([p['action'] for p in candidates])-targets[key][visits[key]],axis=1)))];path='sobol'
            elif rng.random()<.25:proposal=candidates[int(rng.integers(len(candidates)))];path='exploration'
            else:
                sa=torch.tensor([state(p)+p['action'] for p in candidates],dtype=torch.float32)
                with torch.no_grad():score=torch.minimum(agent.q1(sa),agent.q2(sa)).numpy().ravel()
                proposal=candidates[int(np.argmax(score))]
        candidates.remove(proposal);visits[key]+=1
        answer=oracle.query(proposal['id']);row=dict(proposal,**answer);observed.append(row)
        queries.append(dict(row,path=path))
        if agent:
            refresh()
            for _ in range(10):agent.update(32)
    if agent is not None and checkpoint_path is not None:
        from pathlib import Path
        Path(checkpoint_path).parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(actor=agent.actor.state_dict(),q1=agent.q1.state_dict(),q2=agent.q2.state_dict(),
            state_dim=len(public[0]['state'])+18,action_dim=51,seed=seed,method=method,knowledge_features=18,
            scope='terminal contextual acquisition, not an operating controller or risk forecaster'),checkpoint_path)
    return dict(method=method,seed=seed,budget=budget,queries=queries,oracle_calls=list(oracle.calls),
        acquisition_training_calls=int(agent is not None),risk_forecaster_training_calls=0,old_models_training_calls=0,
        algorithm='terminal gamma0 twin-critic finite-pool ranking; boundary hybrid alternates observed-bracket refinement by context')
