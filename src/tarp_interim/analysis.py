from collections import defaultdict
from pathlib import Path
import hashlib
import numpy as np
from scipy.stats import spearmanr
from tarp_boundary.io import read,write
from tarp_sequence.data import load_cases
from tarp_sequence.environment import AcquisitionEnv
from tarp_sequence.baselines import marginal_scores,local_predictions
from .training import regression_metrics


def rank_summary(truth,predicted):
    truth,predicted=np.asarray(truth),np.asarray(predicted)
    if len(truth)<3 or np.ptp(truth)<1e-10:return dict(spearman=None,candidates=len(truth),reason='no discriminating true marginal gains')
    rho=float(spearmanr(truth,predicted).statistic) if np.ptp(predicted)>1e-10 else None
    return dict(spearman=rho,candidates=len(truth),regret=float(truth.max()-truth[np.argmax(predicted)]),
                top1=bool(np.isclose(truth[np.argmax(predicted)],truth.max(),atol=1e-10,rtol=0)))


def true_marginals(env):
    before=env.utility();values=np.full(len(env.available()),np.nan)
    for i in np.where(env.available())[0]:
        env.seen.append(int(i));values[i]=env.utility()-before;env.seen.pop()
    return values


def source_macro(rows,metric):
    by=defaultdict(list)
    for r in rows:
        v=r.get(metric)
        if v is not None and np.isfinite(v):by[r['source_id']].append(v)
    values={s:float(np.mean(v)) for s,v in by.items()}
    return dict(mean=float(np.mean(list(values.values()))) if values else None,sources=values,n_sources=len(values))


def diagnose(root):
    root=Path(root);cases=load_cases(root);response=[];ranking=[]
    for cache in sorted(root.glob('response_s*/predictions.npz')):
        seed=int(cache.parent.name.split('_s')[-1])
        with np.load(cache) as saved:
            for c in cases:
                p=saved[c['key']];refs=np.full(len(c['action']),c['families'].index('reference'))
                m=regression_metrics(p[:,:60],c['response'],c['response_mask'],refs)
                response.append(dict(seed=seed,key=c['key'],source_id=c['source_id'],role=c['role'],**m))
                for extra in [0,4,8]:
                    e=AcquisitionEnv(c,29);available=np.where(e.available())[0]
                    state_seed=int(hashlib.sha256(c['key'].encode()).hexdigest()[:8],16)
                    history=np.random.default_rng(state_seed).permutation(available)[:extra]
                    for i in history:e.step(int(i),score=False)
                    actual=true_marginals(e);eligible=e.available()&c['valid'];ids=np.where(eligible)[0]
                    for method,prediction in [('model',p),('local',local_predictions(c,e,p))]:
                        scores=marginal_scores(c,e,prediction)
                        ranking.append(dict(seed=seed,key=c['key'],source_id=c['source_id'],role=c['role'],history_extra=extra,method=method,
                                            **rank_summary(actual[ids],scores[ids])))
    summary={}
    for role in sorted({c['role'] for c in cases}):
        own=[r for r in response if r['role']==role]
        summary[role]={m:source_macro(own,m) for m in ['mae','zero_mae','delta_mae','delta_zero_mae','direction_accuracy']}
    rank_summary_by={role:{m:source_macro([r for r in ranking if r['role']==role and r['method']==m],'spearman') for m in ['model','local']} for role in summary}
    result=dict(response_rows=response,ranking_rows=ranking,response_summary=summary,ranking_summary=rank_summary_by,
                original_confirm_used=0,scope='Predetermined development holdouts; ranking is immediate marginal utility, not long-horizon optimal value')
    write(root/'diagnostics.json',result);return result
