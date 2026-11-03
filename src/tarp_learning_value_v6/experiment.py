from pathlib import Path
import hashlib
import numpy as np
import torch
from tarp_boundary.io import read,write
from tarp_sequence.policy import SetPolicy,stack_features
from tarp_acq_evidence.pipeline import seal,verify_seal
from .data import role,selected_rows,fingerprint
from .acquisition import Episode,choose_rule
from .policy import policy_logits
from .training import fit_network

METHODS=['value_greedy','value_ppo','local','bracket_random','random']


def selections(cases,encoded,refs,truths,scale,noise,policy_path,method,seed,device):
    model=None
    if method in ['value_ppo','hybrid']:
        model=SetPolicy().to(device);model.load_state_dict(torch.load(policy_path,map_location=device,weights_only=False)['model']);model.eval()
    result=[]
    for c in cases:
        if role(c)!='train':continue
        for budget in [8,16,24]:
            env=Episode(c,encoded[c['key']],refs['train'],scale,noise,budget,seed)
            rng=np.random.default_rng(int(hashlib.sha256(f"{seed}|{c['key']}".encode()).hexdigest()[:8],16))
            before=env.loss(truths['train'])
            while not env.done:
                if method=='greedy_bracket':
                    index=choose_rule(env,'value_greedy' if len(env.queries)%2==0 else 'bracket_random',rng)
                elif model is not None and (method!='hybrid' or len(env.queries)%2==0):
                    with torch.no_grad():logits,_=policy_logits(model,stack_features([env.features()],device))
                    index=int(logits.argmax(-1)[0])
                else:index=choose_rule(env,'bracket_random' if method=='hybrid' else method,rng)
                env.step(index)
            result.append(dict(key=c['key'],source_id=c['source_id'],method=method,seed=seed,budget=budget,
                queries=env.queries,initial_error=before,final_error=env.loss(truths['train'])))
    return result


def fit_students(root,cases,selection_rows,scale,seed,device,epochs):
    root=Path(root);mapping=[];executed=0
    for method in sorted({r['method'] for r in selection_rows}):
        for budget in [8,16,24]:
            queries={r['key']:r['queries'] for r in selection_rows if r['method']==method and r['budget']==budget}
            rows=selected_rows(cases,queries);identity=fingerprint(rows);folder=root/f'students/s{seed}_{identity[:20]}'
            marker=folder/'complete.json'
            if marker.exists():verify_seal(marker)
            else:
                print(f'student {method} s{seed} b{budget}: exposed rows={len(rows)}',flush=True)
                fit_network(cases,rows,folder,scale,seed,device,root/f'priors/s{seed}/model.pt',epochs)
                write(folder/'exposure.json',dict(fingerprint=identity,seed=seed,rows=len(rows),audit_rows=0,test_rows=0))
                seal(marker,[folder/'model.pt',folder/'fit.json',folder/'exposure.json']);executed+=1
            mapping.append(dict(method=method,budget=budget,seed=seed,folder=str(folder.resolve()),fingerprint=identity))
    return mapping,executed


def hybrid_decision(mapping):
    def score(method):
        fits=[read(Path(r['folder'])/'fit.json')['best_validation'] for r in mapping if r['method']==method and r['budget'] in [8,16]]
        if not fits:raise ValueError('Missing tuning fits for hybrid decision')
        return float(np.mean(fits))
    greedy,ppo=score('value_greedy'),score('value_ppo')
    return dict(triggered=bool(ppo>.99*greedy),greedy_tune_error=greedy,ppo_tune_error=ppo,
        rule='PPO on odd queries; verified bracket on even queries, Random if no bracket',
        criterion='PPO has not lowered tuning error by at least 1% over value greedy',test_labels_used=False)
