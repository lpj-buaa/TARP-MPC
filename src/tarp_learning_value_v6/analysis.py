from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.special import expit
from tarp_boundary.io import read,write,sha
from tarp_sequence.data import GROUPS
from tarp_sequence.response import ResponseModel
from tarp_interim.analysis import source_macro
from .data import role
from .core import task_errors
from .training import predict


def case_metrics(raw,case,scale):
    y=case['response'];failure=~case['recoverable'].astype(bool)
    error=task_errors(raw,y,failure,scale);ref=case['families'].index('reference')
    delta=(raw[:,:60]-raw[ref,:60])-(y-y[ref]);significant=np.abs(y-y[ref])>.01
    predicted_failure=raw[:,60:]>0
    positive=int(failure.sum());negative=int((~failure).sum())
    return dict(task_error=float(error.mean()),mae=float(np.abs(raw[:,:60]-y).mean()),
        delta_mae=float(np.abs(delta).mean()),direction_accuracy=float((np.sign(raw[:,:60]-raw[ref,:60])==np.sign(y-y[ref]))[significant].mean()) if significant.any() else None,
        recovery_failure_recall=float((predicted_failure&failure).sum()/positive) if positive else None,
        recovery_failure_fpr=float((predicted_failure&~failure).sum()/negative) if negative else None,
        recovery_failure_positive=positive,recovery_failure_negative=negative,
        **{f'coverage_{e}':float((error<=e).mean()) for e in [.1,.25,.5,1.]},
        groups={g:float(np.abs(raw[:,lo:hi]-y[:,lo:hi]).mean()) for g,(lo,hi) in GROUPS.items()})


def evaluate(root,cases,scale,device):
    root=Path(root)
    if not (root/'models_frozen.json').exists():raise ValueError('Test labels locked until all models freeze')
    freeze=read(root/'models_frozen.json')
    for p,h in freeze['models'].items():
        if sha(p)!=h:raise ValueError('Frozen student changed')
    rows=[];mapping=read(root/'student_manifest.json')['models'];loaded={}
    for item in mapping:
        folder=Path(item['folder']);path=folder/'model.pt'
        if str(path) not in loaded:
            model=ResponseModel().to(device);model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model']);model.eval();loaded[str(path)]=model
        model=loaded[str(path)]
        for c in cases:
            if role(c) not in ['test','development']:continue
            raw=predict(model,[(c,i) for i in range(32)],scale,device)
            rows.append(dict(method=item['method'],budget=item['budget'],seed=item['seed'],key=c['key'],source_id=c['source_id'],role=role(c),**case_metrics(raw,c,scale)))
    for seed in read(root/'run_protocol.json')['seeds']:
        model=ResponseModel().to(device);model.load_state_dict(torch.load(root/f'priors/s{seed}/model.pt',map_location=device,weights_only=False)['model']);model.eval()
        for c in cases:
            if role(c) not in ['test','development']:continue
            raw=predict(model,[(c,i) for i in range(32)],scale,device)
            rows.append(dict(method='initial_only',budget=0,seed=seed,key=c['key'],source_id=c['source_id'],role=role(c),**case_metrics(raw,c,scale)))
    metrics=['task_error','mae','delta_mae','direction_accuracy','recovery_failure_recall','recovery_failure_fpr',*[f'coverage_{e}' for e in [.1,.25,.5,1.]]]
    summary={r:{m:{str(b):{metric:source_macro([x for x in rows if x['role']==r and x['method']==m and x['budget']==b],metric)
        for metric in metrics} for b in sorted({x['budget'] for x in rows if x['method']==m})}
        for m in sorted({x['method'] for x in rows})} for r in ['test','development'] if any(x['role']==r for x in rows)}
    comparisons={}
    for method in sorted({x['method'] for x in rows}-{'initial_only','value_greedy'}):
        comparisons[method]={}
        for b in ['8','16','24']:
            a=summary['test'][method][b]['task_error'];g=summary['test']['value_greedy'][b]['task_error']
            comparisons[method][b]=dict(error_difference=a['mean']-g['mean'],
                relative_reduction_vs_greedy=1-a['mean']/g['mean'],
                paired_sources={s:a['sources'][s]-g['sources'][s] for s in g['sources']})
    b24={seed:{r['fingerprint'] for r in mapping if r['seed']==seed and r['budget']==24} for seed in {r['seed'] for r in mapping}}
    if any(len(v)!=1 for v in b24.values()):raise ValueError('Budget24 data differ across methods')
    priors=[read(p) for p in root.glob('priors/*/fit.json')];policies=[read(p) for p in root.glob('policies/*/fit.json')];students=[read(p) for p in root.glob('students/*/fit.json')]
    fast=[r for p in root.glob('selections/*/rows.json') for r in read(p)['rows']]
    fast=[dict(r,relative_reduction=1-r['final_error']/max(r['initial_error'],1e-8)) for r in fast]
    bootstrap={}
    for method,by_budget in comparisons.items():
        bootstrap[method]={}
        for b,comparison in by_budget.items():
            values=np.array(list(comparison['paired_sources'].values()))
            samples=np.random.default_rng(42).choice(values,(2000,len(values))).mean(1)
            bootstrap[method][b]=dict(error_difference=values.mean(),source_ci95=np.quantile(samples,[.025,.975]).tolist(),sources=len(values))
    result=dict(rows=rows,summary=summary,comparisons=comparisons,source_bootstrap=bootstrap,hybrid=read(root/'hybrid_decision.json'),
        fast_learning_summary={m:{str(b):source_macro([r for r in fast if r['method']==m and r['budget']==b],'relative_reduction') for b in [8,16,24]}
            for m in sorted({r['method'] for r in fast})},
        budget24_same_data_verified=True,test_sources=sorted({c['source_id'] for c in cases if role(c)=='test'}),
        training=dict(prior_fits=len(priors),policy_fits=len(policies),student_fits=len(students),
            prior_seconds=sum(f['seconds'] for f in priors),policy_seconds=sum(f['seconds'] for f in policies),student_seconds=sum(f['seconds'] for f in students),
            network_parameters=sorted({f['parameters'] for f in priors+students}),policy_parameters=sorted({f['parameters'] for f in policies}),
            policy_cache_queries=sum(f['training_cache_queries'] for f in policies),new_simulations=0,old_rl_training=0,old_expert_training=0),
        target='60 physical resource/dispatch outcomes and 3 recovery labels; not four physical risk warning heads',
        test_learner_adaptation=0,scope='Source-held-out reuse; historical cal sources, not newly collected confirm')
    if 'hybrid' in summary['test'] and 'greedy_bracket' in summary['test']:
        result['hybrid_rl_increment']={b:dict(error_difference=summary['test']['hybrid'][b]['task_error']['mean']-
            summary['test']['greedy_bracket'][b]['task_error']['mean'],
            paired_sources={s:summary['test']['hybrid'][b]['task_error']['sources'][s]-v
                for s,v in summary['test']['greedy_bracket'][b]['task_error']['sources'].items()}) for b in ['8','16','24']}
    write(root/'evaluation.json',result)
    return result
