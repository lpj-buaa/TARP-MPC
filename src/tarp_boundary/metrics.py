"""Post-query metrics; full answer sets never enter the acquisition selector."""
import numpy as np


def adjacent_boundaries(rows):
    groups={}
    for r in rows:groups.setdefault((r['key'],r['family']),[]).append(r)
    result=[]
    for group in groups.values():
        group=sorted(group,key=lambda x:x['dose'])
        for a,b in zip(group,group[1:]):
            if a.get('recoverable') is None or b.get('recoverable') is None:continue
            for h,(x,y) in enumerate(zip(a['recoverable'],b['recoverable'])):
                if x!=y:result.append((a['id'],b['id'],h))
    return result


def expanded(rows):
    families=sorted({r['family'] for r in rows if r['family']!='reference'})
    result=[]
    for r in rows:
        if r['family']=='reference':result.extend(dict(r,family=f) for f in families)
        else:result.append(r)
    return result


def evaluate(bundle,result):
    rows=bundle['initial']+[dict(p,**bundle['answers'][p['id']]) for p in bundle['public']]
    truth=expanded(rows);lookup={r['id']:r for r in rows};truth_pairs={(a,b) for a,b,h in adjacent_boundaries(truth)}
    budgets=[0,*[b for b in [8,16,32,64,128] if b<=result['budget']]]
    if result['budget'] not in budgets:budgets.append(result['budget'])
    evaluations=[]
    for budget in budgets:
        observed=bundle['initial']+result['queries'][:budget];seen={r['id'] for r in observed};found={(a,b) for a,b in truth_pairs if a in seen and b in seen}
        per=[]
        for key in sorted({r['key'] for r in rows}):
            full=[r for r in rows if r['key']==key and r.get('descriptor') is not None]
            pool=[dict(p,**bundle['answers'][p['id']]) for p in bundle['public'] if p['key']==key and bundle['answers'][p['id']].get('descriptor') is not None]
            known=[r for r in observed if r['key']==key and r.get('descriptor') is not None]
            own_pairs={(a,b) for a,b in truth_pairs if lookup[a]['key']==key};own_found={(a,b) for a,b in found if lookup[a]['key']==key}
            distance=float(np.linalg.norm(np.array([p['descriptor'] for p in pool])[:,None,:]-np.array([r['descriptor'] for r in known])[None,:,:],axis=-1).min(1).mean()) if pool and known else None
            patterns={tuple(r['turnover']) for r in full};seen_patterns={tuple(r['turnover']) for r in known}
            brackets=[]
            for a,b in own_pairs:
                left,right=lookup[a],lookup[b]
                family=right['family'] if left['family']=='reference' else left['family']
                changed=[h for h,(x,y) in enumerate(zip(left['recoverable'],right['recoverable'])) if x!=y]
                eligible=[(x,y) for x in known for y in known if x['dose']<=left['dose'] and y['dose']>=right['dose'] and x['family'] in [family,'reference'] and y['family'] in [family,'reference'] and any(x['recoverable'][h]!=y['recoverable'][h] for h in changed)]
                brackets.append(min((y['dose']-x['dose'] for x,y in eligible),default=1.))
            per.append(dict(source_id=key,distance=distance,boundary_pairs_found=len(own_found),boundary_pairs_total=len(own_pairs),
                boundary_recall=len(own_found)/len(own_pairs) if own_pairs else None,
                boundary_bracket_width=float(np.mean(brackets)) if brackets else None,
                turnover_coverage=len(patterns&seen_patterns)/len(patterns) if patterns else None))
        mean=lambda name:float(np.mean([r[name] for r in per if r[name] is not None])) if any(r[name] is not None for r in per) else None
        selected=result['queries'][:budget]
        evaluations.append(dict(budget=budget,sources=per,distance=mean('distance'),boundary_recall=mean('boundary_recall'),
            boundary_bracket_width=mean('boundary_bracket_width'),turnover_coverage=mean('turnover_coverage'),
            boundary_pairs_found=len(found),boundary_pairs_total=len(truth_pairs),
            boundary_contexts_found=sum(r['boundary_pairs_found']>0 for r in per),boundary_contexts_total=sum(r['boundary_pairs_total']>0 for r in per),
            joint=sum(all(r.get('event_known',[False]*4)[2:]) and all(r.get('event',[0]*4)[2:]) for r in selected),
            unknown=sum(r.get('descriptor') is None for r in selected)))
    return evaluations
