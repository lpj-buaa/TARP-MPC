"""Separate frozen-policy online adaptation test; global students remain untouched."""
from pathlib import Path
import argparse,hashlib
import numpy as np
import torch
from tarp_boundary.io import read,write,sha
from tarp_sequence.data import load_cases
from tarp_sequence.policy import SetPolicy,stack_features
from tarp_interim.analysis import source_macro
from tarp_learning_value_v6.data import role,select_cases
from tarp_learning_value_v6.learner import load_bank,reward_truth
from tarp_learning_value_v6.acquisition import Episode,choose_rule
from tarp_learning_value_v6.policy import policy_logits


def run(root,device):
    root=Path(root);protocol=read(root/'run_protocol.json')
    if not (root/'models_frozen.json').exists():raise ValueError('Models not frozen')
    cases=select_cases(load_cases(protocol['source']),protocol['mode'])
    scale=np.array(read(root/'exposure_protocol.json')['scale']);truth=reward_truth(cases,'test')
    methods=['value_greedy','value_ppo','local','bracket_random','random']
    if read(root/'hybrid_decision.json')['triggered']:methods+=['hybrid','greedy_bracket']
    rows=[];torch.set_num_threads(1)
    for seed in protocol['seeds']:
        encoded,refs,noise=load_bank(cases,root/f'priors/s{seed}/encoding.npz')
        model=SetPolicy().to(device);model.load_state_dict(torch.load(root/f'policies/s{seed}/model.pt',map_location=device,weights_only=False)['model']);model.eval()
        for method in methods:
            for c in cases:
                if role(c)!='test':continue
                for budget in [8,16,24]:
                    env=Episode(c,encoded[c['key']],refs['test'],scale,noise,budget,seed)
                    rng=np.random.default_rng(int(hashlib.sha256(f"{seed}|{c['key']}".encode()).hexdigest()[:8],16))
                    before=env.loss(truth)
                    while not env.done:
                        ppo=(method=='value_ppo' or method=='hybrid' and len(env.queries)%2==0)
                        if ppo:
                            with torch.no_grad():score,_=policy_logits(model,stack_features([env.features()],device))
                            index=int(score.argmax(-1)[0])
                        else:
                            rule=('bracket_random' if method=='hybrid' else 'value_greedy' if len(env.queries)%2==0 else 'bracket_random') if method in ['hybrid','greedy_bracket'] else method
                            index=choose_rule(env,rule,rng)
                        env.step(index)
                    after=env.loss(truth)
                    rows.append(dict(method=method,seed=seed,key=c['key'],source_id=c['source_id'],budget=budget,
                        initial_error=before,final_error=after,relative_reduction=1-after/max(before,1e-8),queries=env.queries))
        print('local learning test seed '+str(seed)+' complete',flush=True)
    for c in cases:
        for seed in protocol['seeds']:
            values=[r['final_error'] for r in rows if r['key']==c['key'] and r['seed']==seed and r['budget']==24]
            if values and np.ptp(values)>1e-8:raise ValueError('Budget24 set-invariant readout mismatch')
    summary={m:{str(b):{metric:source_macro([r for r in rows if r['method']==m and r['budget']==b],metric)
        for metric in ['initial_error','final_error','relative_reduction']} for b in [8,16,24]} for m in methods}
    write(root/'local_learning_test.json',dict(rows=rows,summary=summary,script_sha=sha(__file__),
        evaluation_sha=sha(root/'evaluation.json'),new_training=0,new_simulations=0,
        scope='Separate online adaptation task on held-out sources: only query labels update temporary readouts; audit labels score only; global student predictions unchanged'))
    print({m:{b:v['final_error']['mean'] for b,v in bs.items()} for m,bs in summary.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(a.root,a.device)
