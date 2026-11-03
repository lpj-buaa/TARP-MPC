from pathlib import Path
import time
import numpy as np
import torch
from tarp_boundary.io import write
from tarp_sequence.data import TRAIN_ROLES,TUNE_ROLES
from tarp_sequence.environment import AcquisitionEnv
from tarp_sequence.policy import SetPolicy,feature_tensors,stack_features,gae,update
from .core import correct_predictions,marginal_teacher,ranking_loss

# Ranking coefficient, observed correction, conditionally trained response prior.
ARMS={'rank':(.2,False,False),'correct_fixed':(0.,True,False),
      'rank_correct_fixed':(.2,True,False),'correct':(0.,True,True),'rank_correct':(.2,True,True)}


def collect(model,cases,predictions,budgets,device,deterministic=False,correction=False,teacher=False):
    if teacher and any(c['role'] not in TRAIN_ROLES for c in cases):
        raise ValueError('Ranking teacher is training-only')
    envs=[AcquisitionEnv(c,min(b,len(c['action'])-len(c['initial']))) for c,b in zip(cases,budgets)]
    histories=[[] for _ in envs]
    for _ in range(max(budgets)):
        active=[i for i,e in enumerate(envs) if not e.done]
        if not active:break
        features=[]
        for i in active:
            p=correct_predictions(cases[i]['action'],predictions[i],envs[i].observations()) if correction else predictions[i]
            features.append(feature_tensors(cases[i],envs[i],p))
        with torch.no_grad():
            logits,values=model(stack_features(features,device));dist=torch.distributions.Categorical(logits=logits)
            actions=logits.argmax(-1) if deterministic else dist.sample();logp=dist.log_prob(actions)
        for slot,i in enumerate(active):
            row=dict(features=features[slot],action=int(actions[slot]),logp=float(logp[slot]),value=float(values[slot]))
            if teacher:
                gain,mask=marginal_teacher(envs[i]);row['gains']=np.pad(gain,(0,32-len(gain)))
                row['eligible']=np.pad(mask,(0,32-len(mask)))
            row['reward']=envs[i].step(row['action'],score=not deterministic)
            histories[i].append(row)
    return histories,[e.metrics() for e in envs]


def update_policy(model,optimizer,histories,device,rank_weight=.2):
    if not rank_weight:return update(model,optimizer,histories,'ppo',device)
    samples=[]
    for h in histories:
        advantages,returns=gae([r['reward'] for r in h],[r['value'] for r in h])
        samples.extend(dict(r,advantage=a,returns=v) for r,a,v in zip(h,advantages,returns))
    features=stack_features([r['features'] for r in samples],device)
    actions=torch.tensor([r['action'] for r in samples],device=device)
    oldlog=torch.tensor([r['logp'] for r in samples],device=device)
    advantage=torch.tensor([r['advantage'] for r in samples],device=device,dtype=torch.float32)
    advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp(min=1e-6)
    returns=torch.tensor([r['returns'] for r in samples],device=device,dtype=torch.float32)
    gains=torch.tensor(np.array([r['gains'] for r in samples]),device=device,dtype=torch.float32)
    eligible=torch.tensor(np.array([r['eligible'] for r in samples]),device=device)
    last={}
    for _ in range(4):
        for ix in torch.randperm(len(samples),device=device).split(128):
            logits,value=model({k:v[ix] for k,v in features.items()});dist=torch.distributions.Categorical(logits=logits)
            ratio=(dist.log_prob(actions[ix])-oldlog[ix]).exp()
            actor=-torch.minimum(ratio*advantage[ix],ratio.clamp(.8,1.2)*advantage[ix]).mean()
            critic=((value-returns[ix])**2).mean();rank=ranking_loss(logits,gains[ix],eligible[ix])
            loss=actor+.5*critic-.01*dist.entropy().mean()+rank_weight*rank
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            last=dict(actor=float(actor.detach()),value=float(critic.detach()),ranking=float(rank.detach()))
    return last


def evaluate_policy(model,cases,predictions,budget,device,correction):
    rows=[]
    for begin in range(0,len(cases),8):
        group=cases[begin:begin+8]
        _,metrics=collect(model,group,[predictions[c['key']] for c in group],[budget]*len(group),device,True,correction)
        rows.extend(dict(source_id=c['source_id'],key=c['key'],role=c['role'],budget=budget,**m) for c,m in zip(group,metrics))
    return rows


def fit_policy(cases,predictions,out,arm,seed=42,device='cuda',batches=120):
    rank_weight,correction,_=ARMS[arm]
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    train=[c for c in cases if c['role'] in TRAIN_ROLES and c['valid'].any()]
    tune=[c for c in cases if c['role'] in TUNE_ROLES and c['valid'].any()]
    if not train or not tune:raise ValueError('Independent training/tuning required')
    model=SetPolicy().to(device);opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    best=-np.inf;history=[];queries=0;episodes=0;steps=0;start=time.perf_counter()
    for batch in range(batches):
        chosen=[train[int(rng.integers(len(train)))] for _ in range(8)]
        budgets=rng.choice([8,16,24],size=8).tolist()
        trajectories,metrics=collect(model,chosen,[predictions[c['key']] for c in chosen],budgets,device,
            correction=correction,teacher=bool(rank_weight))
        losses=update_policy(model,opt,trajectories,device,rank_weight)
        n=sum(len(h) for h in trajectories);queries+=n;episodes+=len(trajectories);steps+=4*int(np.ceil(n/128))
        if (batch+1)%20==0 or batch==0 or batch+1==batches:
            evaluation=evaluate_policy(model,tune,predictions,16,device,correction)
            score=float(np.mean([np.mean([r['utility'] for r in evaluation if r['source_id']==s]) for s in {r['source_id'] for r in evaluation}]))
            history.append(dict(batch=batch+1,tune_utility=score,episode_utility=float(np.mean([m['utility'] for m in metrics])),**losses))
            print(f'{arm} s{seed} batch {batch+1}: tune={score:.4f}, queries={queries}',flush=True)
            if score>best+1e-9:
                best=score;torch.save(dict(model=model.state_dict(),seed=seed,batch=batch+1,arm=arm),out/'model.pt')
    write(out/'fit.json',dict(arm=arm,seed=seed,history=history,parameters=sum(p.numel() for p in model.parameters()),
        episodes=episodes,training_cache_queries=queries,optimizer_steps=steps,policy_training_calls=1,
        rank_weight=rank_weight,correction=correction,elapsed_seconds=time.perf_counter()-start,
        train_sources=sorted({c['source_id'] for c in train}),tune_sources=sorted({c['source_id'] for c in tune}),
        privileged_rank_labels_training_only=bool(rank_weight),new_simulations=0))
