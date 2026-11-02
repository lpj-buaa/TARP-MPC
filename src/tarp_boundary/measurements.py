"""Immutable per-branch MPC measurements, including explicit unknown suffixes."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import time
import numpy as np
from .io import read,write,sha,bind,source_hashes,PROJECT


def _measure(job):
    from tarp37.network import UnifiedNetwork
    from .domain import measure,reference_target
    item,reference,out=job;record=read(item['path']);ref=read(reference['path']);network=UnifiedNetwork();values=[]
    for h in [1,3,5]:
        if len(record['rows'])!=6 or not all(r.get('physical_valid') for r in record['rows']) or len(ref['visible_views'])<=h:
            values.append(dict(known=False,reason='unverified native horizon'));continue
        values.append(measure(network,record['visible_views'][h],reference_target(ref['visible_views'][h])))
    result=dict(id=item['id'],measurements=values,record_hash=sha(item['path']),reference_hash=sha(reference['path']))
    write(out,result);return item['id']


def measure_all(root,workers=4):
    root=Path(root);items=read(root/'inventory.json');refs={x['key']:x for x in items if x['family']=='reference'}
    code=source_hashes(PROJECT/'src/tarp37')
    for name in ['domain.py','measurements.py','io.py']:code[str(PROJECT/'src/tarp_boundary'/name)]=sha(PROJECT/'src/tarp_boundary'/name)
    bind(root/'measurement_protocol.json',dict(code=code,hours=[1,3,5],relative_business_budget=.05,recovery_deadline_hours=1,
        targets='matched reference current ESS reserves at same future time',unknown='any required solver or native horizon unverified'))
    pending=list(items);running={};done=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            for item in list(pending):
                ref=refs[item['key']];path=Path(item['path']);refpath=Path(ref['path']);out=root/'measurements'/(item['id']+'.json')
                if not(path.exists() and refpath.exists()):continue
                if out.exists():
                    value=read(out)
                    assert value['record_hash']==sha(path) and value['reference_hash']==sha(refpath)
                    pending.remove(item);done+=1
                elif len(running)<workers:
                    running[pool.submit(_measure,(item,ref,str(out)))]=item['id'];pending.remove(item)
            if running:
                finished,_=wait(running,timeout=2,return_when=FIRST_COMPLETED)
                for f in finished:
                    ident=f.result();del running[f];done+=1
                    write(root/'measurement_progress.json',dict(completed=done,total=len(items),latest=ident))
                    print(f'MPC profiles {done}/{len(items)}: {ident}',flush=True)
            else:time.sleep(2)
    for file,h in code.items():assert sha(file)==h
    write(root/'measurement_complete.json',dict(complete=True,branches=len(items),states=len(items)*3,
        outputs={str(root/'measurements'/(x['id']+'.json')):sha(root/'measurements'/(x['id']+'.json')) for x in items}))


def make_bundle(root,source):
    from tarp_info.profiles import visible_state
    root=Path(root);items=read(root/'inventory.json')
    refs={x['key']:x for x in items if x['family']=='reference'}
    reference_measures={k:read(root/'measurements'/(x['id']+'.json'))['measurements'] for k,x in refs.items()}
    schema=read(Path(source)/'artifacts/retraining/risk37_td3_v4_s42_20261101/pack_initial/schema.json')
    envelope=np.asarray(schema['station_power_envelope_mw']);action_scale=np.r_[envelope,np.ones(4)]
    physical_ranges=np.array([envelope[:37].sum(),envelope[37:].sum(),8.,60.])
    states=np.array([visible_state(read(x['path'])['context']) for x in refs.values()]);mean=states.mean(0);scale=np.maximum(states.std(0),.01)
    teacher=read(root/'teacher.json');public=[];initial=[];answers={}
    for item in items:
        record=read(item['path']);measure=read(root/'measurements'/(item['id']+'.json'))['measurements'];rm=reference_measures[item['key']]
        known=all(x['known'] for x in measure+rm)
        delta=None;descriptor=None;turnover=None;recoverable=None
        if known:
            delta=(np.array([[x['width_mw'][d] for d in ['SCS','FCS','ESS','PCC']] for x in measure])-
                np.array([[x['width_mw'][d] for d in ['SCS','FCS','ESS','PCC']] for x in rm]))/physical_ranges
            gaps=np.array([x['gap_mwh'] for x in measure])/6.4;recoverable=[x['recoverable'] for x in measure]
            signs=np.where(np.abs(delta)>.01,np.sign(delta),0)
            turnover=np.any(signs>0,axis=0)&np.any(signs<0,axis=0)
            descriptor=(np.r_[delta.ravel()/np.sqrt(12),gaps/np.sqrt(3),np.array(recoverable)/np.sqrt(3),turnover/2]/2).tolist()
        ctx=record['context'];p=dict(id=item['id'],key=item['key'],source_id=item['key'],family=item['family'],dose=item['dose'],
            state=((visible_state(ctx)-mean)/scale).tolist(),action=(np.array(ctx['candidate_mw'])/action_scale).tolist())
        a=dict(descriptor=descriptor,recoverable=recoverable,turnover=turnover.tolist() if turnover is not None else None,
            width_delta=delta.tolist() if delta is not None else None,gap_mwh=[x.get('gap_mwh') for x in measure],error=teacher['errors'][item['id']],
            event=record['event'],event_known=record['event_known'])
        if item['initial']:initial.append(p|a)
        else:public.append(p);answers[p['id']]=a
    bundle=dict(public=public,initial=initial,answers=answers,state_scaler=dict(mean=mean.tolist(),scale=scale.tolist()),
        physical_direction_ranges_mw=physical_ranges.tolist(),scope='V5 native finite-pool acquisition; same six historical initial states; no forecaster training')
    write(root/'bundle.json',bundle);return bundle


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();measure_all(args.root,args.workers)
