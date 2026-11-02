from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
import json,os,time,argparse,shutil
import numpy as np
from tarp_boundary.io import read,write,sha,bind,PROJECT,source_hashes


def context_key(r):return f"{r['source_id']}_h{r['pool_hour']}_p{r['profile_id']}"


def action_bank(base,view,envelope,endpoints=None):
    base=np.asarray(base,float);envelope=np.asarray(envelope,float)
    low=base.copy();low[:47]*=.2
    release=base.copy();release[:37]=-np.minimum(view['discharge_cap_mw'][:37],view['v2g_available_mwh'][:37]);release[47:]=-np.asarray(view['discharge_cap_mw'])[47:]
    if endpoints is not None:low,release=map(lambda x:np.asarray(x,float),endpoints)
    rows=[dict(family='reference',dose=0.,initial=True,power=base.tolist()),
          dict(family='throttle',dose=1.,initial=True,power=low.tolist()),
          dict(family='release',dose=1.,initial=True,power=release.tolist())]
    for family,target in [('throttle',low),('release',release)]:
        for j in range(1,13):rows.append(dict(family=family,dose=j/13,initial=False,power=(base+j/13*(target-base)).tolist()))
    extra=[]
    a=base.copy();a[:47]=release[:47];extra.append(('charging_only',a))
    a=base.copy();a[47:]=release[47:];extra.append(('storage_only',a))
    a=base.copy();order=np.argsort(np.asarray(view['due_hour'])[:37]);donor=order[18:];receiver=order[:18]
    amount=min(float(np.maximum(a[donor],0).sum()*.5),float(np.maximum(envelope[receiver]-a[receiver],0).sum()))
    if amount>0:
        d=np.maximum(a[donor],0);room=np.maximum(envelope[receiver]-a[receiver],0)
        a[donor]-=amount*d/d.sum();a[receiver]+=amount*room/room.sum()
    extra.append(('spatial_shift',a))
    a=base.copy();a[:37]*=.2;a[37:47]=np.minimum(envelope[37:47],base[37:47]+.5*np.asarray(view['charge_cap_mw'])[37:47]);extra.append(('fcs_priority',a))
    a=base.copy();a[:47]=base[:47]+.35*(release[:47]-base[:47]);a[47:]=base[47:]+.8*(release[47:]-base[47:]);extra.append(('asymmetric_joint',a))
    for family,a in extra:rows.append(dict(family=family,dose=1.,initial=False,power=a.tolist()))
    assert len(rows)==32
    return rows


def prepare(root,source,mode='formal'):
    root=Path(root);source=Path(source)
    old=PROJECT/'artifacts/boundary_v2/boundary_formal_s42_v2'
    schema=read(source/'artifacts/retraining/risk37_td3_v4_s42_20261101/pack_initial/schema.json')
    envelope=np.array(schema['station_power_envelope_mw']);contexts=[];inputs={}
    for file in sorted((source/'artifacts/runs/risk37_formal_v4_s42_20261101/sources').glob('*/records.json')):
        inputs[str(file.resolve())]=sha(file);records=read(file)
        for hour in [32,40]:
            for profile in [0,1]:
                own=[r for r in records if r['pool_hour']==hour and r['profile_id']==profile]
                ref=next(r for r in own if r['candidate_id']==0)
                prototype={k:ref[k] for k in ['source_id','role','date','pool_hour','profile_id','context']}
                bank=action_bank(ref['context']['candidate_mw'],ref['context']['view'],envelope,
                                 [next(r for r in own if r['candidate_id']==j)['context']['candidate_mw'] for j in [1,2]])
                key=context_key(ref);items=[]
                for index,a in enumerate(bank):
                    ident=f'{key}_a{index:02d}'
                    items.append(dict(id=ident,key=key,source_id=ref['source_id'],role=ref['role'],index=index,
                                      family=a['family'],dose=a['dose'],initial=a['initial'],power=a['power'],
                                      path=str((root/'contexts'/key/'branches'/ident/'record.json').resolve())))
                contexts.append(dict(key=key,prototype=prototype,items=items))
                snapshot=file.parent/f'pool_{hour}'
                inputs.update({str(p.resolve()):sha(p) for p in snapshot.rglob('*') if p.is_file()})
                if mode=='smoke':break
            if mode=='smoke':break
        if mode=='smoke':break
    if mode=='smoke':
        # A genuine small end-to-end executor check, not a simulated mock.
        contexts[0]['items']=contexts[0]['items'][:4]
    code=source_hashes(PROJECT/'src/tarp37',PROJECT/'src/tarp_info',PROJECT/'src/tarp_boundary')
    code[str(Path(__file__).resolve())]=sha(__file__)
    from tarp37.assets import frozen_inputs
    inputs.update(frozen_inputs())
    for p in (PROJECT/'artifacts/data').glob('*'):
        if p.is_file():inputs[str(p.resolve())]=sha(p)
    bind(root/'physical_protocol.json',dict(inputs=inputs,code=code,source=str(source.resolve()),mode=mode,
         branches_per_context=32,controller='V5 bounded routine Q and unchanged MPC',hours=[1,3,5],old_training_calls=0))
    write(root/'inventory.json',contexts)
    return contexts


def _run_context(task):
    root,source,context=task;root=Path(root);out=root/'contexts'/context['key'];out.mkdir(parents=True,exist_ok=True)
    from tarp_boundary.measurements import _measure
    from tarp37.environment import UnifiedEnvironment,ExecutionUnknown
    from tarp37.risks import window_labels
    from tarp_info.native import regulator
    oldroot=PROJECT/'artifacts/boundary_v2/boundary_formal_s42_v2'
    olditems={r['id']:r for r in read(oldroot/'inventory.json')}
    prototype=context['prototype'];reference=context['items'][0];env=None
    try:
        for item in context['items']:
            folder=Path(item['path']).parent;folder.mkdir(parents=True,exist_ok=True)
            done=folder/'complete.json';measurefile=folder/'measurement.json'
            if done.exists():
                m=read(done)
                assert sha(item['path'])==m['record_sha'] and sha(measurefile)==m['measurement_sha']
                continue
            oldid=None
            if prototype['pool_hour']==40 and prototype['profile_id']==0:
                suffix='reference' if item['index']==0 else f"{item['family']}_13" if item['initial'] else f"{item['family']}_{int(round(item['dose']*13)):02d}"
                guess=prototype['source_id']+'_'+suffix
                if guess in olditems:
                    candidate=read(olditems[guess]['path'])
                    if np.allclose(candidate['context']['candidate_mw'],item['power'],atol=1e-10,rtol=0):oldid=guess
            if not Path(item['path']).exists():
                if oldid:
                    shutil.copyfile(olditems[oldid]['path'],item['path'])
                    write(folder/'reuse.json',dict(record=olditems[oldid]['path'],sha=sha(item['path']),old_id=oldid))
                else:
                    if env is None:env=UnifiedEnvironment(out/'simulation',prototype['date'],int(prototype['source_id'].rsplit('_s',1)[1]));env.regulator=regulator(env.network)
                    env.restore(Path(source)/'artifacts/runs/risk37_formal_v4_s42_20261101/sources'/prototype['source_id']/f"pool_{prototype['pool_hour']}")
                    env.anchor=np.array(prototype['context']['preference_anchor']);ctx=env.context();ctx['candidate_mw']=item['power']
                    rows=[];views=[]
                    with (folder/'native.log').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
                        for h in range(6):
                            views.append(env.context());plan=env.plan()
                            try:rows.append(env.step(np.array(item['power']) if h==0 else plan['power_mw'],plan))
                            except ExecutionUnknown as error:rows.append(error.row);break
                            print(f'hour {h+1}/6',flush=True)
                    event,known=window_labels(rows,contract=env.c)
                    value={k:prototype[k] for k in ['source_id','role','date','pool_hour','profile_id']}
                    value.update(candidate_id=item['id'],family=item['family'],context=ctx,rows=rows,visible_views=views,
                                 event=event.tolist(),event_known=known.tolist(),hardware=env.c.manifest(),
                                 origin='snapshot-restored one intervention; frozen V5 six-hour MPC continuation')
                    write(item['path'],value)
            if not measurefile.exists():
                oldmeasure=oldroot/'measurements'/(oldid+'.json') if oldid else None
                if oldmeasure is not None and oldmeasure.exists():
                    value=read(oldmeasure)
                    assert value['record_hash']==sha(item['path']) and value['reference_hash']==sha(reference['path'])
                    value['id']=item['id'];write(measurefile,value)
                else:_measure((item,reference,str(measurefile)))
            write(done,dict(record_sha=sha(item['path']),measurement_sha=sha(measurefile),reused=bool(oldid)))
            write(out/'progress.json',dict(completed=sum((Path(i['path']).parent/'complete.json').exists() for i in context['items']),total=len(context['items']),latest=item['id']))
    finally:
        if env is not None:env.close()
    write(out/'complete.json',dict(complete=True,items=len(context['items']),outputs={str(Path(i['path']).parent/'complete.json'):sha(Path(i['path']).parent/'complete.json') for i in context['items']}))
    return context['key']


def collect(root,source,mode='formal',workers=4,resume=False):
    root=Path(root)
    if root.exists() and not resume:raise ValueError('RunId exists: use --resume')
    root.mkdir(parents=True,exist_ok=True)
    contexts=prepare(root,source,mode);done=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(_run_context,(str(root),str(source),c)):c['key'] for c in contexts}
        for f in as_completed(futures):
            key=f.result();done+=1
            write(root/'collection_progress.json',dict(contexts_complete=done,total_contexts=len(contexts),latest=key))
            print(f'context {done}/{len(contexts)} complete: {key}',flush=True)
    proto=read(root/'physical_protocol.json')
    for p,h in {**proto['inputs'],**proto['code']}.items():assert sha(p)==h,p
    items=[i for c in contexts for i in c['items']]
    reused=sum(read(Path(i['path']).parent/'complete.json')['reused'] for i in items)
    write(root/'collection_complete.json',dict(complete=True,contexts=len(contexts),branches=len(items),reused=reused,
        new_native=len(items)-reused,new_measurements=len(items)-reused,old_model_training=0,physics_frozen=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--mode',default='formal',choices=['formal','smoke']);p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    a=p.parse_args();collect(PROJECT/'artifacts/sequence_v4'/a.run_id,PROJECT.parent/'tarp_37_risk_audit',a.mode,a.workers,a.resume)


if __name__=='__main__':main()
