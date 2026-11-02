"""Pure reads: never import a simulator, controller or LP solver here."""
from pathlib import Path
import numpy as np
from tarp_boundary.io import read,sha,write
from .features import spatial_raw,SPATIAL_GROUPS

HOURS=[1,3,5]
DIRECTIONS=['SCS','FCS','ESS','PCC']
RESPONSE_GROUPS={'bounds':(0,24),'reserve_effort':(24,36),'gap':(36,39),'followup':(39,48),'preference':(48,60)}


def stable_response(record,reference,measure,refmeasure,ranges):
    bounds=(np.array([[m['bounds_mw'][d] for d in DIRECTIONS] for m in measure])-
            np.array([[m['bounds_mw'][d] for d in DIRECTIONS] for m in refmeasure]))/np.asarray(ranges)[None,:,None]
    views=[record['visible_views'][h] for h in HOURS];refs=[reference['visible_views'][h] for h in HOURS]
    debt=(np.array([v['view']['ess_energy_mwh'] for v in refs])-
          np.array([v['view']['ess_energy_mwh'] for v in views]))/.95
    gaps=np.array([m['gap_mwh'] for m in measure])/6.4
    def powers(r):
        a=np.array([r['rows'][h]['candidate_request_mw'] for h in HOURS])
        return np.stack([a[:,:37].sum(1),a[:,37:47].sum(1),a[:,47:51].sum(1)],axis=1)
    follow=(powers(record)-powers(reference))/np.array([ranges[0],ranges[1],4.])
    weights=np.array([v['weights'] for v in views])-np.array([v['weights'] for v in refs])
    result=np.r_[bounds.ravel(),debt.ravel(),gaps,follow.ravel(),weights.ravel()]
    assert result.shape==(60,) and np.all(np.isfinite(result))
    return result.tolist()


def frozen_files(source,project):
    files=set(source.glob('*.json'))
    for folder in ['branches','measurements','acquisition','baselines']:
        files.update((source/folder).rglob('*.json'))
        files.update((source/folder).rglob('*.pt'))
    for pkg in ['tarp37','tarp_info','tarp_boundary']:
        files.update((project/'src'/pkg).rglob('*.py'))
    # Existing protocols list external physical inputs, RL and teacher checkpoints.
    def paths(value):
        if isinstance(value,dict):
            for k,v in value.items():
                yield k;yield from paths(v)
        elif isinstance(value,list):
            for v in value:yield from paths(v)
        elif isinstance(value,str):yield value
    for proto in source.glob('*protocol.json'):
        for value in paths(read(proto)):
            if len(value)<300 and ('/' in value or '\\' in value):
                p=Path(value)
                try:
                    if p.is_file():files.add(p)
                except OSError:pass
    return {str(p.resolve()):sha(p) for p in sorted(files)}


def prepare(source,out):
    source=Path(source);bundle=read(source/'bundle.json');items=read(source/'inventory.json')
    records={i['id']:read(i['path']) for i in items}
    measurements={i['id']:read(source/'measurements'/(i['id']+'.json')) for i in items}
    refs={i['key']:i['id'] for i in items if i['family']=='reference'}
    for i in items:
        m=measurements[i['id']]
        assert sha(i['path'])==m['record_hash']
        ref=next(j for j in items if j['id']==refs[i['key']])
        assert sha(ref['path'])==m['reference_hash']
        assert all(v['known'] for v in m['measurements'])
    # Context is one pre-intervention state, not tiny branch-dependent round-off.
    raw=np.array([spatial_raw(records[ident]['context']) for ident in refs.values()])
    mean=raw.mean(0);scale=np.maximum(raw.std(0),.01)
    group_weight=np.concatenate([np.full(n,np.sqrt(n)) for n in SPATIAL_GROUPS])
    public=[];initial=[];answers={};initial_lookup={r['id']:r for r in bundle['initial']}
    for p in bundle['public']+bundle['initial']:
        ident=p['id'];ref=refs[p['key']];ctx=records[ref]['context']
        r={k:p[k] for k in ['id','key','source_id','family','dose','state','action']}
        r['spatial']=((spatial_raw(ctx)-mean)/scale/group_weight).tolist()
        base=initial_lookup.get(ident,bundle['answers'].get(ident))
        a={k:base[k] for k in ['descriptor','recoverable','turnover','width_delta','gap_mwh','error','event','event_known']}
        a['response']=stable_response(records[ident],records[ref],measurements[ident]['measurements'],
                                      measurements[ref]['measurements'],bundle['physical_direction_ranges_mw'])
        if ident in initial_lookup:initial.append(r|a)
        else:public.append(r);answers[ident]=a
    result=dict(public=public,initial=initial,answers=answers,spatial_scaler=dict(mean=mean.tolist(),scale=scale.tolist()),
                response_groups=RESPONSE_GROUPS,physical_direction_ranges_mw=bundle['physical_direction_ranges_mw'],
                scope='162 frozen V5 branches; no new native or LP calls; retrospective same-six-source finite pool')
    assert len(initial)==18 and len(public)==144 and len(answers)==144
    write(out,result);return result
