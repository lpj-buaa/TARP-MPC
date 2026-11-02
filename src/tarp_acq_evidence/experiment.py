from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from tarp_boundary.learning import information_reward
from .features import state_input,memory

VARIANTS=['old_summary','quantitative','stable','combined']


def from_response(x,response):
    r=np.asarray(response);delta=(r[:24].reshape(3,4,2)[:,:,1]-r[:24].reshape(3,4,2)[:,:,0])
    debt=r[24:36].reshape(3,4)
    # One-hour 1 MW charge-rate, eta .95. A necessary resource boundary; exact for this cache.
    gaps=.95*np.maximum(debt-1.,0).sum(1)
    recover=gaps<=1e-6
    signs=np.where(np.abs(delta)>.01,np.sign(delta),0)
    turn=np.any(signs>0,axis=0)&np.any(signs<0,axis=0)
    descriptor=np.r_[delta.ravel()/np.sqrt(12),gaps/6.4/np.sqrt(3),recover/np.sqrt(3),turn/2]/2
    return dict(x,response=r.tolist(),descriptor=descriptor.tolist(),recoverable=recover.tolist(),turnover=turn.tolist())


def interpolate(x,observed):
    family=[r for r in observed if r['key']==x['key'] and r['id']!=x['id'] and r['family'] in [x['family'],'reference']]
    lower=max((r for r in family if r['dose']<=x['dose']),key=lambda r:r['dose'],default=None)
    upper=min((r for r in family if r['dose']>=x['dose']),key=lambda r:r['dose'],default=None)
    if lower is None or upper is None:raise ValueError('Known endpoints required for local interpolation')
    fraction=(x['dose']-lower['dose'])/max(upper['dose']-lower['dose'],1e-12)
    return from_response(x,(1-fraction)*np.array(lower['response'])+fraction*np.array(upper['response']))


def ranking_metrics(candidate_ids,scores,truth,keys):
    result=[]
    for key in sorted(set(keys)):
        ix=np.where(np.array(keys)==key)[0];pred=np.array(scores)[ix];y=np.array(truth)[ix]
        rho=(float(spearmanr(pred,y).statistic) if np.ptp(pred)>1e-12 else 0.) if len(ix)>1 and np.ptp(y)>1e-12 else None
        n=min(3,len(ix));top=np.argsort(-pred,kind='stable')[:n];optimal=np.sort(y)[-n:]
        utility=float(y[top].sum());oracle=float(optimal.sum())
        result.append(dict(source_id=key,spearman=rho,top3_utility=utility/n,oracle_top3_utility=oracle/n,
                           top3_capture=utility/oracle if oracle>1e-12 else None,regret=(oracle-utility)/n,n=len(ix)))
    return result


def experiment(public,initial,oracle,variant,budget,seed,fixed=None,warm=100,updates=10,
               checkpoints=(16,32,64,128),checkpoint_dir=None):
    from .model import Agent
    if variant not in VARIANTS+['local_interpolation']:raise ValueError(variant)
    quantitative=variant in ['quantitative','combined'];stable=variant in ['stable','combined']
    agent=None if variant=='local_interpolation' else Agent(seed,stable)
    rng=np.random.default_rng(seed);observed=list(initial);remaining=list(public);keys=sorted({p['key'] for p in public})
    queries=[];rankings=[];training_steps=0
    def inputs(rows):
        return ([state_input(x,quantitative) for x in rows],[x['action'] for x in rows],
                [memory(x,observed,quantitative) for x in rows])
    def train(steps):
        nonlocal training_steps
        agent.train_steps(*inputs(observed),[information_reward(x,observed) for x in observed],
                          [x['response'] for x in observed],steps)
        training_steps+=steps
    def predict(rows):
        if agent is not None:return agent.predict(*inputs(rows))
        predictions=[interpolate(x,observed) for x in rows]
        return np.array([information_reward(x,observed) for x in predictions]),np.array([x['response'] for x in predictions])
    if agent:train(warm)
    for i in range(budget):
        candidates=[p for p in remaining if p['key']==keys[i%len(keys)]]
        if fixed is not None:proposal=next(x for x in remaining if x['id']==fixed[i]);path='matched-history'
        elif rng.random()<.25:proposal=candidates[int(rng.integers(len(candidates)))];path='exploration'
        else:
            scores,_=predict(candidates);proposal=candidates[int(np.argmax(scores))];path='rank'
        remaining.remove(proposal);answer=oracle.query(proposal['id']);observed.append(dict(proposal,**answer))
        queries.append(dict(observed[-1],path=path))
        if agent:train(updates)
        if i+1 in checkpoints:
            scores,response=predict(remaining)
            rankings.append(dict(budget=i+1,candidate_ids=[x['id'] for x in remaining],keys=[x['key'] for x in remaining],
                                 scores=scores.tolist(),response_predictions=response.tolist()))
            if agent is not None and checkpoint_dir is not None:
                import torch
                folder=Path(checkpoint_dir);folder.mkdir(parents=True,exist_ok=True)
                torch.save(dict(net=agent.net.state_dict(),variant=variant,budget=i+1,seed=seed),folder/f'b{i+1}.pt')
    return dict(method=variant,seed=seed,budget=budget,queries=queries,oracle_calls=list(oracle.calls),rankings=rankings,
                acquisition_training_calls=int(agent is not None),training_steps=training_steps,
                native_simulations=0,mpc_solves=0,old_rl_training_calls=0,forecaster_training_calls=0,expert_training_calls=0)
