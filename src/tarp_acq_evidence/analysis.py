from pathlib import Path
from collections import defaultdict
import numpy as np
from tarp_boundary.io import read,write


def source_values(records,method):
    grouped=defaultdict(list)
    for r in records:
        if r['method']==method and r['value'] is not None:grouped[r['source_id']].append(r['value'])
    return {k:float(np.mean(v)) for k,v in grouped.items()}


def paired(records,left,right,draws=2000):
    a=source_values(records,left);b=source_values(records,right);keys=sorted(a.keys()&b.keys())
    values=np.array([a[k]-b[k] for k in keys]);rng=np.random.default_rng(20261102)
    samples=values[rng.integers(len(values),size=(draws,len(values)))].mean(1)
    return dict(left=left,right=right,difference=float(values.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),
                n_sources=len(keys),source_differences={k:float(v) for k,v in zip(keys,values)})


def summary(records):
    table={}
    for method in sorted({r['method'] for r in records}):
        values=source_values(records,method)
        seeds={s:float(np.mean([r['value'] for r in records if r['method']==method and r['seed']==s and r['value'] is not None]))
               for s in sorted({r['seed'] for r in records if r['method']==method})}
        table[method]=dict(mean=float(np.mean(list(values.values()))),sources=values,seeds=seeds,n_sources=len(values))
    return table


def analyze(root):
    root=Path(root);runs={str(p.relative_to(root/'experiments')):read(p) for p in (root/'experiments').glob('*/*/result.json')}
    ranking=defaultdict(list);active=defaultdict(list);errors=defaultdict(list)
    for label,r in runs.items():
        protocol=label.split('/')[0]
        if protocol=='matched':
            for snap in r['rankings']:
                for source in snap['sources']:
                    for metric in ['spearman','top3_capture','regret','top3_utility']:
                        ranking[(snap['budget'],metric)].append(dict(method=r['method'],seed=r['seed'],source_id=source['source_id'],value=source[metric]))
                if r['method'] in ['stable','combined']:
                    for source in snap['derived_sources']:
                        ranking[(snap['budget'],'derived_spearman')].append(dict(method=r['method'],seed=r['seed'],source_id=source['source_id'],value=source['spearman']))
                    for group,values in snap['response_errors'].items():
                        for k,value in values.items():errors[(snap['budget'],group)].append(dict(method=r['method'],seed=r['seed'],source_id=k,value=value))
        if protocol=='active':
            for snap in r['evaluation']:
                for source in snap['sources']:
                    for metric in ['distance','boundary_recall','boundary_bracket_width','turnover_coverage']:
                        active[(snap['budget'],metric)].append(dict(method=r['method'],seed=r['seed'],source_id=source['source_id'],value=source[metric]))
    for label,evaluations in read(root/'baselines.json').items():
        method,seed=label.rsplit('_s',1)
        for snap in evaluations:
            for source in snap['sources']:
                for metric in ['distance','boundary_recall','boundary_bracket_width','turnover_coverage']:
                    active[(snap['budget'],metric)].append(dict(method=method,seed=int(seed),source_id=source['source_id'],value=source[metric]))
    report=dict(ranking={f'{b}/{m}':summary(v) for (b,m),v in ranking.items()},
                active={f'{b}/{m}':summary(v) for (b,m),v in active.items()},
                response_mae={f'{b}/{m}':summary(v) for (b,m),v in errors.items()},comparisons={})
    comparisons=[('quantitative','old_summary'),('stable','old_summary'),('combined','old_summary'),('combined','quantitative'),
                 ('combined','stable'),('combined','local_interpolation')]
    for (budget,metric),records in ranking.items():
        if metric=='derived_spearman':continue
        report['comparisons'][f'ranking/{budget}/{metric}']=[paired(records,a,b) for a,b in comparisons]
    for (budget,metric),records in active.items():
        report['comparisons'][f'active/{budget}/{metric}']=[paired(records,a,b) for a,b in comparisons+[('combined','random'),('combined','bracket_random')]]
    write(root/'analysis.json',report)
    write(root/'source_records.json',dict(ranking={f'{b}/{m}':v for (b,m),v in ranking.items()},
                                        active={f'{b}/{m}':v for (b,m),v in active.items()}))
    return report
