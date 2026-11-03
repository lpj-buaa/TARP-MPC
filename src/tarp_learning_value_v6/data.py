import hashlib
import numpy as np
from tarp_sequence.data import GROUPS


def role(case):
    return {'train':'train','meta_train':'train','tune':'tune','meta_tune':'tune',
        'cal':'test','development_test':'development'}[case['role']]


def partition(case):
    initial=list(case['initial']);eligible=[i for i in range(len(case['action'])) if i not in initial]
    bins=[[i for i in eligible if case['families'][i]==f and (case['doses'][i]<.5)==low]
        for f in ['throttle','release'] for low in [True,False]]
    bins.append([i for i in eligible if case['families'][i] not in ['throttle','release']])
    def key(i):return hashlib.sha256(f"lv6|{case['key']}|{case['ids'][i]}".encode()).hexdigest()
    if any(not b for b in bins):raise ValueError('Audit stratification requires the complete 32-action pool')
    audit=sorted(min(b,key=key) for b in bins)
    query=[i for i in eligible if i not in audit]
    if len(initial)!=3 or len(audit)!=5 or len(query)!=24:raise ValueError('Expected 3 initial, 5 audit, 24 query')
    return dict(initial=initial,audit=audit,query=query)


def select_cases(cases,mode):
    if mode!='smoke':return cases
    chosen={}
    for r,limit in [('train',2),('tune',1),('test',1),('development',1)]:
        chosen[r]=sorted({c['source_id'] for c in cases if role(c)==r})[:limit]
    return [c for c in cases if c['source_id'] in chosen[role(c)]]


def response_scale(cases):
    rows=np.concatenate([c['response'][c['initial']] for c in cases if role(c)=='train'])
    scale=np.zeros(60)
    for lo,hi in GROUPS.values():scale[lo:hi]=max(float(np.sqrt(np.mean(rows[:,lo:hi]**2))),.02)
    return scale


def public(case):
    return {k:case[k] for k in ['key','source_id','state','action','families','doses','ids','initial']}


def selected_rows(cases,selections=None):
    rows=[]
    for c in cases:
        if role(c)!='train':continue
        p=partition(c);indices=sorted(set(p['initial']+(selections[c['key']] if selections is not None else [])))
        if set(indices)&set(p['audit']):raise ValueError('Audit targets cannot fit the learner')
        rows.extend((c,int(i)) for i in indices)
    return rows


def fingerprint(rows):
    identities=sorted(f"{c['key']}|{c['ids'][i]}" for c,i in rows)
    return hashlib.sha256('\n'.join(identities).encode()).hexdigest()
