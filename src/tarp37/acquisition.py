"""TD3-style one-intervention acquisition, followed by fixed-policy MPC.

This is a terminal one-decision MDP (gamma=0): RL chooses a probing action,
not the six-hour operating policy. The frozen initial student defines learning
utility; independent meta/cal/confirm trajectories never become RL rewards.
"""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import os
import numpy as np
import torch
from torch import nn
from .assets import PROJECT,PYTHON,sha
from .collection import read,write,verify,stamp
from .packing import load_packed,transform_context,raw_targets,INPUTS
from .model import ConsequenceStudent
from .learning import resolve_device


class AcquisitionTD3:
    def __init__(self,state_dim,device='cpu',seed=42):
        torch.manual_seed(seed); self.device=device; self.rng=np.random.default_rng(seed); self.replay=[]; self.steps=0
        self.actor=nn.Sequential(nn.Linear(state_dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,51),nn.Tanh()).to(device)
        def critic(): return nn.Sequential(nn.Linear(state_dim+51,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1)).to(device)
        self.q1,self.q2=critic(),critic()
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=.0003)
        self.q_optimizer=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=.0003)

    def action(self,state,noise=.15):
        with torch.no_grad(): value=self.actor(torch.as_tensor(state,device=self.device,dtype=torch.float32)).cpu().numpy()
        return np.clip(value+self.rng.normal(0,noise,51),-1,1)

    def add(self,state,action,reward): self.replay.append((np.array(state,np.float32),np.array(action,np.float32),float(reward)))

    def update(self,batch_size=64):
        if len(self.replay)<4: return None
        idx=self.rng.integers(0,len(self.replay),min(batch_size,len(self.replay)))
        state=torch.tensor(np.stack([self.replay[i][0] for i in idx]),device=self.device)
        action=torch.tensor(np.stack([self.replay[i][1] for i in idx]),device=self.device)
        reward=torch.tensor([[self.replay[i][2]] for i in idx],device=self.device)
        sa=torch.cat([state,action],1); loss=(self.q1(sa)-reward).square().mean()+(self.q2(sa)-reward).square().mean()
        self.q_optimizer.zero_grad(set_to_none=True); loss.backward(); self.q_optimizer.step(); self.steps+=1
        if self.steps%2==0:
            actor_loss=-self.q1(torch.cat([state,self.actor(state)],1)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward(); self.actor_optimizer.step()
        return float(loss.detach())


def action_bounds(ctx,hardware):
    v=ctx['view']; hi=np.r_[np.minimum(hardware,np.maximum(v['charge_cap_mw'][:47],np.asarray(v['arrival_prior_mw'])[0])*1.5),v['charge_cap_mw'][47:]]
    lo=np.r_[np.minimum(np.asarray(v['discharge_cap_mw'])[:37],np.asarray(v['v2g_available_mwh'])[:37]),np.zeros(10),v['discharge_cap_mw'][47:]]
    return hi,lo


def physical_action(action,ctx,hardware):
    hi,lo=action_bounds(ctx,hardware); a=np.asarray(action)
    p=np.where(a>=0,a*hi,a*lo)
    p[37:47]=(a[37:47]+1)*.5*hi[37:47]
    return p


def actor_state(ctx,scalers):
    f=transform_context(ctx,np.zeros(51),scalers)
    return np.concatenate([f[k].reshape(-1) for k in ['nodes','history','forecast','context','initial_resource','initial_response','initial_weight']]).astype(np.float32)


def acquire(data_run,run_id,config,device='auto',policy='td3',episodes=12,resume=False):
    root=PROJECT/'artifacts/retraining'/run_id; out=root/'explore'; marker=out/'complete.json'
    if marker.exists(): verify(read(marker)); print('explore: reused verified queries',flush=True); return
    if out.exists() and not resume: raise ValueError('Acquisition exists; use Resume or a new RunId')
    out.mkdir(parents=True,exist_ok=True); queries=out/'queries'; proposals=out/'proposals'; queries.mkdir(exist_ok=True); proposals.mkdir(exist_ok=True)
    packed=load_packed(root/'pack_initial'); checkpoint=root/'fit_initial/checkpoint.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False); device=resolve_device(device); torch.set_num_threads(1)
    model=ConsequenceStudent(saved['schema'],saved['decoder'],saved['width']).to(device); model.load_state_dict(saved['state_dict']); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    records=[]
    for p in sorted((PROJECT/'artifacts/runs'/data_run/'sources').glob('*/records.json')): records+=read(p)
    reference=[r for r in records if r['role']=='train' and r['candidate_id']==0]
    if not reference: raise ValueError('Training-source native pools are required')
    hardware=packed['schema']['station_power_envelope_mw']; seed=int(config.get('seed',42)); rng=np.random.default_rng(seed)
    agent=AcquisitionTD3(len(actor_state(reference[0]['context'],packed['scalers'])),device,seed)
    references={(r['source_id'],r['pool_hour'],r['profile_id']):r for r in reference}; seen=set(); logs=[]
    def predict(record):
        features=transform_context(record['context'],record['context']['candidate_mw'],packed['scalers'])
        with torch.no_grad(): return model({k:torch.as_tensor(v,device=device) for k,v in features.items()})['outcome'].cpu().numpy()[0]
    ref_predictions={key:predict(r) for key,r in references.items()}
    def utility(record):
        key=(record['source_id'],record['pool_hour'],record['profile_id']); ref=references[key]
        t=raw_targets(record); rt=raw_targets(ref); mask=t['outcome_mask']&rt['outcome_mask']
        scale=np.array(packed['scalers']['outcome']['scale'])
        delta=(t['outcome']-rt['outcome'])/scale
        predicted=predict(record)-ref_predictions[key]
        error=float(np.mean(np.abs(predicted[mask]-delta[mask]))) if mask.any() else 0.
        signature=tuple(record['event']); novel=signature not in seen; seen.add(signature)
        return min(error,5.)+.25*novel-.25*(1-mask.mean()),error
    # Seed replay uses existing training branches only; no extra simulator calls.
    for record in records:
        if record['role']!='train': continue
        key=(record['source_id'],record['pool_hour'],record['profile_id'])
        if key not in references: continue
        reward,_=utility(record); hi,lo=action_bounds(record['context'],hardware); power=np.array(record['context']['candidate_mw'])
        action=np.divide(power,np.where(power>=0,hi,lo),out=np.zeros(51),where=np.where(power>=0,hi,lo)>1e-9)
        action[37:47]=np.divide(power[37:47],hi[37:47],out=np.zeros(10),where=hi[37:47]>1e-9)*2-1
        agent.add(actor_state(record['context'],packed['scalers']),np.clip(action,-1,1),reward)
    for _ in range(200): agent.update()
    from scipy.stats import qmc
    points=qmc.Sobol(51,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(max(episodes,1)))))[:episodes]*2-1
    dependencies=[checkpoint,Path(__file__),PROJECT/'src/tarp37/query.py',PROJECT/'src/tarp37/model.py',
                  PROJECT/'src/tarp37/packing.py',root/'pack_initial/complete.json',root/'fit_initial/complete.json']
    inputs={str(p):sha(p) for p in dependencies}
    # Freeze the exact saved native states actually queried, in addition to the
    # compact published trajectory records.
    for r in reference:
        pool=PROJECT/'artifacts/runs'/data_run/'sources'/r['source_id']/f"pool_{r['pool_hour']}"
        for p in pool.rglob('*'):
            if p.is_file(): inputs[str(p)]=sha(p)
    for k in range(episodes):
        ref=reference[k%len(reference)]; state=actor_state(ref['context'],packed['scalers'])
        action=agent.action(state) if policy=='td3' else points[k] if policy=='sobol' else rng.uniform(-1,1,51)
        proposal=proposals/f'query_{k:05d}.json'; result=queries/f'query_{k:05d}.json'
        if not result.exists():
            public_reference={key:ref[key] for key in ['source_id','role','date','pool_hour','profile_id','context']}
            write(proposal,dict(reference=public_reference,power_mw=physical_action(action,ref['context'],hardware).tolist(),candidate_id=f'{policy}_{k:05d}',family=policy+'_acquisition'))
            process=subprocess.run([str(PYTHON),'-B','-u','-m','tarp37.query','--data-run',data_run,'--proposal',str(proposal),'--output',str(result)],
                cwd=PROJECT,env={**os.environ,'PYTHONPATH':str(PROJECT/'src')},capture_output=True,text=True,encoding='utf-8',errors='replace')
            (out/f'query_{k:05d}.log').write_text(process.stdout+'\n'+process.stderr,encoding='utf-8')
            if process.returncode: raise RuntimeError('Native acquisition query failed; inspect '+str(out/f'query_{k:05d}.log'))
        record=read(result); reward,error=utility(record)
        # An interrupted run reuses the action actually queried, not a newly
        # sampled actor action associated with an old cached reward.
        executed=np.array(record['context']['candidate_mw']); hi,lo=action_bounds(ref['context'],hardware)
        action=np.divide(executed,np.where(executed>=0,hi,lo),out=np.zeros(51),where=np.where(executed>=0,hi,lo)>1e-9)
        action[37:47]=np.divide(executed[37:47],hi[37:47],out=np.zeros(10),where=hi[37:47]>1e-9)*2-1
        action=np.clip(action,-1,1)
        if policy=='td3':
            agent.add(state,action,reward)
            for _ in range(50): agent.update()
        logs.append(dict(query=k,source_id=ref['source_id'],utility=reward,paired_prediction_error=error))
        print(f'{policy} acquisition {k+1}/{episodes}: utility={reward:.4f}',flush=True)
    torch.save(dict(actor=agent.actor.state_dict(),q1=agent.q1.state_dict(),q2=agent.q2.state_dict(),
                    state_dim=len(state),schema='one_intervention_terminal_acquisition_gamma0',policy=policy),out/'acquisition.pt')
    write(out/'queries_summary.json',logs)
    verify(dict(inputs=inputs,outputs={}))
    stamp(out,inputs,policy=policy,episodes=episodes,new_acquisition_training_calls=int(policy=='td3'),
          old_rl_training_calls=0,old_expert_training_calls=0)
