import numpy as np
from scipy.stats import qmc
from tarp_boundary.learning import information_reward
from .response import predicted_descriptor


def known_rows(case,env):
    result=[]
    for i,r,m,recovery,k in env.observations():
        if not m.all() or not k.all():continue
        p=np.r_[r,~recovery.astype(bool)][None];desc=predicted_descriptor(p)[0]
        # Actual verified recovery labels take precedence over the predictor's necessary-bound projection.
        desc[15:18]=np.asarray(recovery)/np.sqrt(3)/2
        result.append(dict(index=i,id=case['ids'][i],key=case['key'],family=case['families'][i],dose=case['doses'][i],
                           action=case['action'][i],response=r,recoverable=np.asarray(recovery).tolist(),
                           descriptor=desc,turnover=(desc[-4:]>0).tolist()))
    return result


def local_predictions(case,env,prediction):
    known=known_rows(case,env);result=np.zeros_like(prediction);result[:,60:63]=.5
    if not known:return result
    for i,a in enumerate(case['action']):
        family=case['families'][i];dose=case['doses'][i]
        own=[r for r in known if r['family'] in [family,'reference']]
        left=max((r for r in own if r['dose']<=dose),key=lambda r:r['dose'],default=None)
        right=min((r for r in own if r['dose']>=dose),key=lambda r:r['dose'],default=None)
        if left is not None and right is not None:
            t=(dose-left['dose'])/max(right['dose']-left['dose'],1e-12);rows=[left,right];weights=np.array([1-t,t])
        else:
            rows=sorted(known,key=lambda r:np.linalg.norm(a-r['action']))[:3]
            d=np.array([np.linalg.norm(a-r['action']) for r in rows]);weights=1/np.maximum(d,1e-6);weights/=weights.sum()
        result[i,:60]=np.sum([w*r['response'] for w,r in zip(weights,rows)],axis=0)
        # Interpolate the continuous LP excess over the necessary energy/rate bound,
        # rather than imposing a spurious 50% boundary between binary endpoint labels.
        def lower_bound(r):return .95*np.maximum(np.asarray(r)[24:36].reshape(3,4)-1.,0.).sum(1)/6.4
        residual=np.sum([w*np.maximum(r['response'][36:39]-lower_bound(r['response']),0.) for w,r in zip(weights,rows)],axis=0)
        result[i,36:39]=lower_bound(result[i])+residual
        result[i,60:63]=(result[i,36:39]*6.4>1e-6).astype(float)
    return result


def scores(case,env,prediction):
    known=known_rows(case,env);descriptors=predicted_descriptor(prediction);result=[]
    for i,p in enumerate(prediction):
        x=dict(id=case['ids'][i],key=case['key'],family=case['families'][i],dose=case['doses'][i],descriptor=descriptors[i],
               recoverable=(descriptors[i,15:18]>.1).tolist(),turnover=(descriptors[i,-4:]>0).tolist())
        result.append(information_reward(x,known))
    return np.array(result)


def marginal_scores(case,env,prediction):
    """Predicted immediate gain of the SAME terminal utility, without hidden answers."""
    from .environment import AcquisitionEnv
    from .data import boundary_pairs
    descriptors=predicted_descriptor(prediction);recovery=descriptors[:,15:18]>.1
    valid=np.ones(len(descriptors),bool)
    known=known_rows(case,env);known_indices={r['index'] for r in known}
    for r in known:descriptors[r['index']]=r['descriptor'];recovery[r['index']]=r['recoverable']
    for i in env.seen:
        if i not in known_indices:valid[i]=False
    items=[dict(family=f,dose=d) for f,d in zip(case['families'],case['doses'])]
    surrogate=dict(descriptor=descriptors,valid=valid,recoverable=recovery,initial=case['initial'],boundary_pairs=boundary_pairs(items,valid,recovery))
    estimate=AcquisitionEnv(surrogate,max(env.remaining_budget,1));estimate.seen=list(env.seen)
    before=estimate.utility();values=np.full(len(descriptors),-np.inf)
    for i in np.flatnonzero(env.available()):
        estimate.seen.append(int(i));values[i]=estimate.utility()-before;estimate.seen.pop()
    return values


def select(case,env,prediction,method,rng,sobol=None):
    available=np.flatnonzero(env.available())
    if method=='random':return int(rng.choice(available))
    if method=='sobol':
        families=sorted(set(case['families'])-{'reference'});point=sobol[len(env.queries)]
        coords=np.array([[families.index(case['families'][i])/max(len(families)-1,1),case['doses'][i]] for i in available])
        return int(available[np.linalg.norm(coords-point,axis=1).argmin()])
    known=known_rows(case,env)
    if method=='bracket_random' and len(env.queries)%2==1:
        brackets=[]
        for family in ['throttle','release']:
            rows=sorted([r for r in known if r['family'] in [family,'reference']],key=lambda r:r['dose'])
            for left,right in zip(rows,rows[1:]):
                if left['recoverable']==right['recoverable']:continue
                inside=[i for i in available if case['families'][i]==family and left['dose']<case['doses'][i]<right['dose']]
                if inside:brackets.append((right['dose']-left['dose'],min(inside,key=lambda i:abs(case['doses'][i]-(left['dose']+right['dose'])/2))))
        if brackets:return int(max(brackets)[1])
    if method=='bracket_random':return int(rng.choice(available))
    values=local_predictions(case,env,prediction) if method=='local' else prediction
    rank=marginal_scores(case,env,values)
    return int(available[np.argmax(rank[available])])


def run(case,prediction,method,budget,seed):
    from .environment import AcquisitionEnv
    env=AcquisitionEnv(case,min(budget,len(case['action'])-len(case['initial'])));rng=np.random.default_rng(seed)
    points=qmc.Sobol(2,scramble=True,seed=seed).random_base2(5)
    while not env.done:env.step(select(case,env,prediction,method,rng,points),score=False)
    return env.metrics()
