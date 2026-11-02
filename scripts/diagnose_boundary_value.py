"""Evaluate a budget-64 critic on answers it had not observed at budget 64."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from tarp_boundary.io import read,write,PROJECT
from tarp_boundary.learning import knowledge_features,information_reward
from tarp37.acquisition import AcquisitionTD3


def correlation(a,b):
    if len(a)<3 or np.std(a)<1e-10 or np.std(b)<1e-10:return None
    return float(spearmanr(a,b).statistic)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    root=PROJECT/'artifacts/boundary_v2'/args.run_id;bundle=read(root/'bundle.json');torch.set_num_threads(1);records=[]
    for method in ['error_td3','boundary_reward_only','boundary_td3']:
        for seed in [42,43,44]:
            run=read(root/'acquisition'/f'{method}_s{seed}.json');observed=bundle['initial']+run['queries'][:64];seen={r['id'] for r in observed}
            remaining=[p for p in bundle['public'] if p['id'] not in seen]
            saved=torch.load(root/'acquisition'/f'{method}_s{seed}_b64.pt',map_location='cpu',weights_only=False)
            model=AcquisitionTD3(saved['state_dim'],'cpu',seed);model.q1.load_state_dict(saved['q1']);model.q2.load_state_dict(saved['q2'])
            by={}
            for p in remaining:
                tensor=torch.tensor([p['state']+knowledge_features(p,observed)+p['action']],dtype=torch.float32)
                with torch.no_grad():score=float(torch.minimum(model.q1(tensor),model.q2(tensor)).item())
                actual=dict(p,**bundle['answers'][p['id']]);utility=information_reward(actual,observed)
                own=min(actual['error'],5.) if method=='error_td3' else utility
                by.setdefault(p['key'],[]).append(dict(id=p['id'],score=score,own_utility=own,information_utility=utility))
            for source,rows in by.items():records.append(dict(method=method,seed=seed,source_id=source,unseen_at_budget64=len(rows),
                own_rho=correlation([r['score'] for r in rows],[r['own_utility'] for r in rows]),
                information_rho=correlation([r['score'] for r in rows],[r['information_utility'] for r in rows]),predictions=rows))
    summary=[]
    for method in ['error_td3','boundary_reward_only','boundary_td3']:
        mean=lambda key:float(np.mean([r[key] for r in records if r['method']==method and r[key] is not None])) if any(r['method']==method and r[key] is not None for r in records) else None
        summary.append(dict(method=method,own_rho=mean('own_rho'),information_rho=mean('information_rho')))
    write(root/'value_diagnostics.json',dict(complete=True,budget=64,scope='post-query finite-pool ranking on 80 answers unobserved at budget64; no refit',summary=summary,records=records,training_calls=0))
    print('Budget64 held-out-answer value ranking:',summary)


if __name__=='__main__':main()
