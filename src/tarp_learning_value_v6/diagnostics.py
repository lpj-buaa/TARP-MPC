from pathlib import Path
import hashlib
import numpy as np
import torch
from scipy.stats import spearmanr
from tarp_boundary.io import read,write
from tarp_sequence.policy import SetPolicy,stack_features
from tarp_interim.analysis import source_macro
from .data import role,partition
from .core import targets,raw_prediction,task_errors
from .learner import load_bank,reward_truth
from .acquisition import Episode
from .policy import policy_logits


def feature_variants(features,oracle_values):
    result={name:{k:np.array(v,copy=True) for k,v in features.items()} for name in
        ['without_candidate_outcomes','without_observed_outcomes','without_information','oracle_information']}
    result['without_candidate_outcomes']['candidates'][:,51:114]=0
    result['without_observed_outcomes']['observations'][:,51:111]=0
    result['without_observed_outcomes']['observations'][:,171:174]=0
    result['without_information']['information'][:]=0
    result['without_information']['candidates'][:,114:116]=0
    result['oracle_information']['information']=np.asarray(oracle_values,dtype=np.float32)
    result['oracle_information']['candidates'][:,114]=oracle_values/max(float(np.max(np.abs(oracle_values))),1e-4)
    return result


def run_diagnostics(root,cases,scale,device):
    root=Path(root)
    if not (root/'models_frozen.json').exists():raise ValueError('Diagnostics locked until model freeze')
    points=[];sensitivity=[];checkpoints={}
    for seed in read(root/'run_protocol.json')['seeds']:
        encoded,refs,noise=load_bank(cases,root/f'priors/s{seed}/encoding.npz')
        truth=reward_truth(cases,'test');model=SetPolicy().to(device)
        checkpoint=torch.load(root/f'policies/s{seed}/model.pt',map_location=device,weights_only=False)
        checkpoints[str(seed)]=int(checkpoint['batch']);model.load_state_dict(checkpoint['model']);model.eval()
        for c in cases:
            if role(c)!='test':continue
            for extra in [0,4,8]:
                env=Episode(c,encoded[c['key']],refs['test'],scale,noise,16,seed)
                key_seed=int(hashlib.sha256(c['key'].encode()).hexdigest()[:8],16)
                for i in np.random.default_rng(key_seed).permutation(partition(c)['query'])[:extra]:env.step(int(i))
                available=np.flatnonzero(env.available());estimated=env.values()[0];actual=np.zeros(32);before=env.loss(truth)
                for i in available:
                    learner=env.learner.copy();y=targets(c['response'][i:i+1],~c['recoverable'][i:i+1].astype(bool),scale)[0]
                    learner.observe(int(i),y)
                    errors=task_errors(raw_prediction(learner.mean[32:],scale),truth['response'],truth['failure'],scale)
                    after=np.mean([errors[np.array(truth['sources'])==s].mean() for s in set(truth['sources'])])
                    actual[i]=before-after
                identity=dict(seed=seed,source_id=c['source_id'],key=c['key'],history_extra=extra)
                rho=float(spearmanr(estimated[available],actual[available]).statistic) if np.ptp(estimated[available])>1e-12 and np.ptp(actual[available])>1e-12 else None
                points.append(dict(identity,spearman=rho,marginal_mae=float(np.abs(estimated[available]-actual[available]).mean()),
                    negative_actual_fraction=float((actual[available]<0).mean()),
                    top1_regret=float(actual[available].max()-actual[available[np.argmax(estimated[available])]])))
                f=env.features()
                with torch.no_grad():logits,_=policy_logits(model,stack_features([f],device))
                original=int(logits.argmax(-1)[0])
                for variant,v in feature_variants(f,actual).items():
                    with torch.no_grad():logits,_=policy_logits(model,stack_features([v],device))
                    chosen=int(logits.argmax(-1)[0])
                    sensitivity.append(dict(identity,variant=variant,choice_changed=float(chosen!=original),
                        actual_gain_change=float(actual[chosen]-actual[original])))
    write(root/'diagnostics.json',dict(value_rows=points,sensitivity_rows=sensitivity,ppo_best_batches=checkpoints,
        value_summary={m:source_macro(points,m) for m in ['spearman','marginal_mae','negative_actual_fraction','top1_regret']},
        sensitivity_summary={v:{m:source_macro([r for r in sensitivity if r['variant']==v],m) for m in ['choice_changed','actual_gain_change']}
            for v in sorted({r['variant'] for r in sensitivity})},
        scope='Post-freeze independent-source diagnostics; oracle information never selects training data or checkpoints'))
