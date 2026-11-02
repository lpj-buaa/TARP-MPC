from pathlib import Path
import time
import numpy as np
import torch
from tarp_boundary.io import read,write
from .data import TRAIN_ROLES,TUNE_ROLES
from .policy import SetPolicy,trajectories,update


def evaluate_policy(model,cases,predictions,budget,device):
    results=[]
    for begin in range(0,len(cases),8):
        rows=cases[begin:begin+8];_,metrics=trajectories(model,rows,[predictions[c['key']] for c in rows],[min(budget,len(c['action'])-len(c['initial'])) for c in rows],device,True)
        for c,m in zip(rows,metrics):results.append(dict(source_id=c['source_id'],key=c['key'],role=c['role'],budget=budget,**m))
    return results


def fit_policy(cases,predictions,out,method,seed=42,device='cpu',smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    train=[c for c in cases if c['role'] in TRAIN_ROLES and c['valid'].any()]
    tune=[c for c in cases if c['role'] in TUNE_ROLES and c['valid'].any()]
    if not tune:
        if not smoke:raise ValueError('No tuning contexts')
        tune=train
    if not train:raise ValueError('No verified training contexts')
    model=SetPolicy().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);history=[];best=-np.inf;query_count=0;episode_count=0
    start=time.perf_counter()
    for batch in range(3 if smoke else 120):
        if method=='group':
            chosen=[train[int(rng.integers(len(train)))]]*4+[train[int(rng.integers(len(train)))]]*4
            budgets=[int(rng.choice([8,16,24]))]*4+[int(rng.choice([8,16,24]))]*4
        else:
            chosen=[train[int(rng.integers(len(train)))] for _ in range(8)];budgets=rng.choice([8,16,24],size=8).tolist()
        histories,metrics=trajectories(model,chosen,[predictions[c['key']] for c in chosen],budgets,device)
        losses=update(model,optimizer,histories,method,device);query_count+=sum(len(h) for h in histories);episode_count+=len(histories)
        if (batch+1)%20==0 or smoke or batch==0:
            evaluation=evaluate_policy(model,tune,predictions,16,device)
            # Same number of contexts per source; explicit macro avoids future imbalance.
            groups={s:[r['utility'] for r in evaluation if r['source_id']==s] for s in {r['source_id'] for r in evaluation}}
            score=float(np.mean([np.mean(v) for v in groups.values()]))
            history.append(dict(batch=batch+1,tune_utility=score,episode_utility=float(np.mean([m['utility'] for m in metrics])),**losses))
            print(f'{method} s{seed} batch {batch+1}: tune={score:.4f}, queries={query_count}',flush=True)
            if score>best+1e-9:
                best=score;torch.save(dict(model=model.state_dict(),method=method,seed=seed,batch=batch+1,gamma=0 if method=='myopic' else 1),out/'model.pt')
    write(out/'fit.json',dict(method=method,seed=seed,history=history,episodes=episode_count,training_cache_queries=query_count,
        policy_training_calls=1,old_model_training_calls=0,elapsed_seconds=time.perf_counter()-start,
        train_sources=sorted({c['source_id'] for c in train}),tune_sources=sorted({c['source_id'] for c in tune}),smoke_resubstitution=bool(smoke)))
    return model


def evaluate(root,cases,seeds,device='cpu',smoke=False):
    from .baselines import run
    root=Path(root);rows=[];timings=[]
    evaluable=[c for c in cases if (c['role'] not in TRAIN_ROLES or smoke) and c['valid'].any()]
    for seed in seeds:
        with np.load(root/f'response_s{seed}/predictions.npz') as cache:predictions={k:cache[k] for k in cache.files}
        for method in ['ppo','myopic','group']:
            model=SetPolicy().to(device);checkpoint=torch.load(root/f'policies/{method}_s{seed}/model.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['model']);model.eval()
            for budget in [8,16,24]:
                begin=time.perf_counter();result=evaluate_policy(model,evaluable,predictions,budget,device)
                rows.extend(dict(r,method=method,seed=seed) for r in result);timings.append(dict(method=method,seed=seed,budget=budget,seconds=time.perf_counter()-begin))
        for method in ['random','sobol','local','bracket_random','greedy']:
            for budget in [8,16,24]:
                for c in evaluable:
                    result=run(c,predictions[c['key']],method,budget,seed)
                    rows.append(dict(source_id=c['source_id'],key=c['key'],role=c['role'],budget=budget,method=method,seed=seed,**result))
    write(root/'evaluation.json',dict(rows=rows,timings=timings,excluded_contexts=[dict(key=c['key'],role=c['role'],reason='no full verified response descriptor') for c in cases if not c['valid'].any()],
                                   confirm_never_used_for_updates=True,new_simulations_this_stage=0))
    return rows
