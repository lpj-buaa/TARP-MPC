from pathlib import Path
import numpy as np
from tarp_boundary.io import read,write,sha,PROJECT
from tarp_info.profiles import visible_state
from tarp_acq_evidence.features import spatial_raw,SPATIAL_GROUPS

TRAIN_ROLES={'train','meta_train'}
TUNE_ROLES={'tune','meta_tune'}
GROUPS={'bounds':(0,24),'reserve_effort':(24,36),'gap':(36,39),'followup':(39,48),'preference':(48,60)}


def targets(record,reference,measure,refmeasure,ranges):
    y=np.zeros(60);mask=np.zeros(60,bool);gaps=np.zeros(3);known=np.zeros(3,bool);width=np.zeros((3,4))
    physical=len(record['rows'])==6 and all(r.get('physical_valid') for r in record['rows'])
    refphysical=len(reference['rows'])==6 and all(r.get('physical_valid') for r in reference['rows'])
    for t,h in enumerate([1,3,5]):
        m,rm=measure[t],refmeasure[t]
        if m['known'] and rm['known']:
            b=(np.array([m['bounds_mw'][d] for d in ['SCS','FCS','ESS','PCC']])-
               np.array([rm['bounds_mw'][d] for d in ['SCS','FCS','ESS','PCC']]))/np.array(ranges)[:,None]
            y[t*8:t*8+8]=b.ravel();mask[t*8:t*8+8]=True;width[t]=b[:,1]-b[:,0]
            y[36+t]=m['gap_mwh']/6.4;mask[36+t]=True;gaps[t]=m['gap_mwh'];known[t]=True
        if len(record['visible_views'])>h and len(reference['visible_views'])>h:
            v=record['visible_views'][h];r=reference['visible_views'][h]
            y[24+t*4:28+t*4]=(np.array(r['view']['ess_energy_mwh'])-v['view']['ess_energy_mwh'])/.95
            mask[24+t*4:28+t*4]=True;y[48+t*4:52+t*4]=np.array(v['weights'])-r['weights'];mask[48+t*4:52+t*4]=True
        if physical and refphysical:
            a=np.array(record['rows'][h]['candidate_request_mw'])-reference['rows'][h]['candidate_request_mw']
            y[39+t*3:42+t*3]=[a[:37].sum()/ranges[0],a[37:47].sum()/ranges[1],a[47:].sum()/4.];mask[39+t*3:42+t*3]=True
    valid=bool(known.all() and physical and refphysical)
    recover=gaps<=1e-6
    signs=np.where(np.abs(width)>.01,np.sign(width),0);turn=(signs>0).any(0)&(signs<0).any(0)
    descriptor=np.r_[width.ravel()/np.sqrt(12),gaps/6.4/np.sqrt(3),recover/np.sqrt(3),turn/2]/2 if valid else np.zeros(22)
    return dict(response=y,response_mask=mask,gaps=gaps,recoverable=recover,recovery_mask=known,descriptor=descriptor,valid=valid)


def boundary_pairs(items,valid,recover):
    result=[]
    for family in ['throttle','release']:
        seq=sorted([i for i,r in enumerate(items) if r['family'] in ['reference',family]],key=lambda i:items[i]['dose'])
        for a,b in zip(seq,seq[1:]):
            if valid[a] and valid[b] and np.any(recover[a]!=recover[b]):result.append((a,b))
    return sorted(set(result))


def pack(root):
    root=Path(root);contexts=read(root/'inventory.json');raw=[];cases=[]
    from tarp37.contract import PhysicalContract
    contract=PhysicalContract.load();envelope=np.array(contract.station_power_envelope_mw);scale=np.r_[envelope,np.ones(4)]
    ranges=[envelope[:37].sum(),envelope[37:].sum(),8.,60.]
    for c in contexts:
        items=c['items'];records=[read(i['path']) for i in items]
        measurements=[read(Path(i['path']).parent/'measurement.json') for i in items]
        for item,m in zip(items,measurements):
            assert m['record_hash']==sha(item['path']) and m['reference_hash']==sha(items[0]['path'])
        ref=records[0];ctx=ref['context'];raw.append(np.r_[visible_state(ctx),spatial_raw(ctx)])
        ts=[targets(r,ref,m['measurements'],measurements[0]['measurements'],ranges) for r,m in zip(records,measurements)]
        case={k:np.array([t[k] for t in ts]) for k in ts[0]}
        case.update(key=c['key'],source_id=items[0]['source_id'],role=items[0]['role'],families=[i['family'] for i in items],
                    doses=[i['dose'] for i in items],initial=[i for i,x in enumerate(items) if x['initial']],
                    action=np.array([i['power'] for i in items])/scale,ids=[i['id'] for i in items])
        case['boundary_pairs']=boundary_pairs(items,case['valid'],case['recoverable']);cases.append(case)
    train=np.array([c['role'] in TRAIN_ROLES for c in cases]);raw=np.array(raw)
    mean=raw[train].mean(0);std=np.maximum(raw[train].std(0),.01)
    weights=np.r_[np.full(49,7.),np.concatenate([np.full(n,np.sqrt(n)) for n in SPATIAL_GROUPS])]
    for case,x in zip(cases,raw):case['state']=(x-mean)/std/weights
    def serial(c):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in c.items()}
    result=dict(cases=[serial(c) for c in cases],scaler=dict(mean=mean.tolist(),scale=std.tolist(),training_sources=sorted({c['source_id'] for c in cases if c['role'] in TRAIN_ROLES})),
                groups=GROUPS,scope='V5 native expanded bank; original role separation; unknown labels masked',
                counts=dict(contexts=len(cases),branches=sum(len(c['ids']) for c in cases),known_branches=sum(int(c['valid'].sum()) for c in cases),
                            boundary_pairs=sum(len(c['boundary_pairs']) for c in cases)))
    write(root/'dataset.json',result);return result


def load_cases(root):
    data=read(Path(root)/'dataset.json');cases=[]
    arraykeys=['state','action','response','response_mask','gaps','recoverable','recovery_mask','descriptor','valid']
    for c in data['cases']:
        cases.append(dict(c,**{k:np.array(c[k]) for k in arraykeys}))
    return cases
