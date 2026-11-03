"""Post-freeze diagnostic; never changes a policy, selected sample, or learner fit."""
from pathlib import Path
import argparse,hashlib
import numpy as np
from scipy.stats import spearmanr
from tarp_boundary.io import read,write,sha
from tarp_sequence.data import load_cases
from tarp_interim.analysis import source_macro
from tarp_learning_value_v6.data import role,partition,select_cases
from tarp_learning_value_v6.core import virtual_values,targets,raw_prediction,task_errors
from tarp_learning_value_v6.learner import load_bank,reward_truth
from tarp_learning_value_v6.acquisition import Episode


def run(root):
    root=Path(root);protocol=read(root/'run_protocol.json')
    if not (root/'models_frozen.json').exists():raise ValueError('Models not frozen')
    cases=select_cases(load_cases(protocol['source']),protocol['mode']);scale=np.array(read(root/'exposure_protocol.json')['scale'])
    truth=reward_truth(cases,'test');rows=[]
    for seed in protocol['seeds']:
        encoded,refs,noise=load_bank(cases,root/f'priors/s{seed}/encoding.npz')
        for c in cases:
            if role(c)!='test':continue
            for extra in [0,4,8]:
                env=Episode(c,encoded[c['key']],refs['test'],scale,noise,16,seed)
                key_seed=int(hashlib.sha256(c['key'].encode()).hexdigest()[:8],16)
                for i in np.random.default_rng(key_seed).permutation(partition(c)['query'])[:extra]:env.step(int(i))
                ix=np.flatnonzero(env.available());actual=[];before=env.loss(truth)
                for i in ix:
                    learner=env.learner.copy();learner.observe(int(i),targets(c['response'][i:i+1],~c['recoverable'][i:i+1].astype(bool),scale)[0])
                    error=task_errors(raw_prediction(learner.mean[32:],scale),truth['response'],truth['failure'],scale)
                    after=np.mean([error[np.array(truth['sources'])==s].mean() for s in set(truth['sources'])]);actual.append(before-after)
                actual=np.array(actual);draw_seed=int(hashlib.sha256(f"{seed}|{c['key']}|{sorted(env.seen)}".encode()).hexdigest()[:8],16)
                scores=[]
                for j in range(16):
                    score,_=virtual_values(env.learner,ix,32+np.array(refs['test']['probe']),noise,draw_seed+j*100003)
                    scores.append(score)
                altered=env.learner.copy();altered.mean[:,:60]=0
                changed,_=virtual_values(altered,ix,32+np.array(refs['test']['probe']),noise,draw_seed)
                identity=dict(seed=seed,source_id=c['source_id'],key=c['key'],history_extra=extra,
                    continuous_mean_invariance_max=float(np.max(np.abs(changed-scores[0]))))
                for n in [1,4,16]:
                    score=np.mean(scores[:n],axis=0)
                    rho=float(spearmanr(score,actual).statistic) if np.ptp(score)>1e-12 and np.ptp(actual)>1e-12 else None
                    rows.append(dict(identity,function_draws=n*4,spearman=rho,marginal_mae=float(np.abs(score-actual).mean()),
                        top1_regret=float(actual.max()-actual[score.argmax()])))
    write(root/'value_estimation_probe.json',dict(rows=rows,summary={str(n):{m:source_macro([r for r in rows if r['function_draws']==n],m)
        for m in ['spearman','marginal_mae','top1_regret']} for n in [4,16,64]},
        continuous_mean_invariance_max=max(r['continuous_mean_invariance_max'] for r in rows),
        evaluation_sha=sha(root/'evaluation.json'),script_sha=sha(__file__),
        scope='Post-hoc value-estimator diagnosis only; no selection or model changed',new_simulations=0,new_training=0))
    print(read(root/'value_estimation_probe.json')['summary'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
