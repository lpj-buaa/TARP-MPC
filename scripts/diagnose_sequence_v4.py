from pathlib import Path
from collections import defaultdict
import argparse,json
import numpy as np


def binary_summary(labels,probability,known):
    y=np.asarray(labels,bool)[known];p=np.asarray(probability)[known];pred=p>=.5
    positive=int(y.sum());negative=int((~y).sum());tp=int((pred&y).sum());fp=int((pred&~y).sum());ece=0.
    for lo in np.arange(0,1,.1):
        mask=(p>=lo)&(p<lo+.1 if lo<.9 else p<=1.)
        if mask.any():ece+=mask.mean()*abs(p[mask].mean()-y[mask].mean())
    return dict(known=int(len(y)),positive=positive,negative=negative,tp=tp,fp=fp,
                recall=tp/positive if positive else None,precision=tp/(tp+fp) if tp+fp else None,
                fpr=fp/negative if negative else None,brier=float(np.mean((p-y)**2)) if len(y) else None,
                zero_brier=float(y.mean()) if len(y) else None,ece=float(ece) if len(y) else None)


def diagnose(root):
    root=Path(root);data=json.loads((root/'dataset.json').read_text('utf-8'));cases=data['cases'];rows=[];recovery=[]
    groups={'bounds':(0,24),'reserve_effort':(24,36),'gap':(36,39),'followup':(39,48),'preference':(48,60)}
    for cachepath in root.glob('response_s*/predictions.npz'):
        seed=int(cachepath.parent.name.split('_s')[-1])
        with np.load(cachepath) as predictions:
            for c in cases:
                if c['role'] in ['train','meta_train']:continue
                y=np.array(c['response']);m=np.array(c['response_mask'],bool);p=predictions[c['key']]
                for name,(lo,hi) in groups.items():
                    valid=m[:,lo:hi]
                    if not valid.any():continue
                    rows.append(dict(seed=seed,source_id=c['source_id'],role=c['role'],key=c['key'],group=name,known=int(valid.sum()),
                        mae=float(np.abs(p[:,lo:hi]-y[:,lo:hi])[valid].mean()),zero_mae=float(np.abs(y[:,lo:hi])[valid].mean())))
                for h in range(3):
                    result=binary_summary(np.array(c['gaps'])[:,h]>1e-6,p[:,60+h],np.array(c['recovery_mask'],bool)[:,h])
                    recovery.append(dict(seed=seed,source_id=c['source_id'],role=c['role'],key=c['key'],hour=[1,3,5][h],**result))
    summary={}
    for role in sorted({r['role'] for r in rows}):
        summary[role]={}
        for group in groups:
            own=[r for r in rows if r['role']==role and r['group']==group];by=defaultdict(list)
            for r in own:by[r['source_id']].append(r)
            if by:summary[role][group]={metric:float(np.mean([np.mean([r[metric] for r in rs]) for rs in by.values()])) for metric in ['mae','zero_mae']}
    cal=[c for c in cases if c['role']=='cal'];support=dict(sources=len({c['source_id'] for c in cal}),contexts=len(cal),
        verified_branches=sum(sum(c['valid']) for c in cal),total_branches=sum(len(c['ids']) for c in cal),
        recovery_positive=sum(int((np.array(c['gaps'])>1e-6)[np.array(c['recovery_mask'],bool)].sum()) for c in cal),
        recovery_negative=sum(int((np.array(c['gaps'])<=1e-6)[np.array(c['recovery_mask'],bool)].sum()) for c in cal),
        fitted_parameters=0,threshold=.5,scope='frozen support/Brier/ECE checks; no calibration fitting or model selection')
    result=dict(response_summary=summary,response_rows=rows,recovery_rows=recovery,calibration_support=support,
                confirm_never_updates_parameters=True,extra_training_calls=0)
    (root/'response_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=diagnose(a.root);print(json.dumps(r['calibration_support'],ensure_ascii=False))
