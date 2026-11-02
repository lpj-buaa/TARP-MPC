"""Native V5 dose branches, sharing immutable old physical initial states."""
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import numpy as np
from .io import read,write,sha,bind,source_hashes,PROJECT


def inventory(source,mode='formal'):
    result=[]
    for f in sorted((Path(source)/'artifacts/runs/risk37_formal_v4_s42_20261101/sources').glob('*/records.json')):
        rows=[r for r in read(f) if r['role']=='train' and r['pool_hour']==40 and r['profile_id']==0]
        if not rows:continue
        ref=next(r for r in rows if r['candidate_id']==0)
        key=ref['source_id'];base=np.array(ref['context']['candidate_mw'])
        for family,cid in [('reference',0),('throttle',1),('release',2)]:
            r=next(r for r in rows if r['candidate_id']==cid)
            result.append(dict(id=f'{key}_{family}_13' if cid else f'{key}_reference',key=key,family=family,dose=1. if cid else 0.,
                initial=True,record=r,input_record_file=str(f),old_candidate_id=cid))
        for family,cid in [('throttle',1),('release',2)]:
            target=np.array(next(r for r in rows if r['candidate_id']==cid)['context']['candidate_mw'])
            for j in range(1,13):
                ident=f'{key}_{family}_{j:02d}';r=deepcopy(ref)
                r['candidate_id']=ident;r['family']=family+'_dose';r['context']['candidate_mw']=(base+j/13*(target-base)).tolist()
                result.append(dict(id=ident,key=key,family=family,dose=j/13,initial=False,record=r,input_record_file=str(f),old_candidate_id=None))
        if mode=='smoke':break
    return result


def _native(task):
    from tarp_info.native import native_job
    source,item,folder=task;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'native.log').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
        return native_job((source,item['record'],str(folder)))


def collect(root,source,workers=4,mode='formal'):
    root=Path(root);items=inventory(source,mode)
    dependencies=source_hashes(PROJECT/'src/tarp37',PROJECT/'src/tarp_info')
    dependencies.update({str(Path(__file__).resolve()):sha(__file__),str(PROJECT/'src/tarp_boundary/io.py'):sha(PROJECT/'src/tarp_boundary/io.py')})
    sources={x['input_record_file']:sha(x['input_record_file']) for x in items}
    for key in {x['key'] for x in items}:
        state=Path(source)/'artifacts/runs/risk37_formal_v4_s42_20261101/sources'/key/'pool_40'
        sources.update({str(f.resolve()):sha(f) for f in state.rglob('*') if f.is_file()})
    bind(root/'collection_protocol.json',dict(mode=mode,code=dependencies,source_inputs=sources,grid='two families lambda j/13 j1..12; endpoints public initial',
        physics='V5 bounded routine Q; original capacities/thresholds',old_models_training=0))
    entries=[];jobs=[];reused=0
    previous=PROJECT/'artifacts/native_information/v5_information_sanity_v1'
    for item in items:
        folder=root/'branches'/item['id'];path=folder/'record.json'
        old=previous/f"{item['key']}_a{item['old_candidate_id']}"/'record.json'
        if not path.exists() and item['initial'] and old.exists():
            value=read(old)
            assert np.allclose(value['context']['candidate_mw'],item['record']['context']['candidate_mw'],rtol=0,atol=1e-9)
            write(path,value);write(folder/'reuse.json',dict(source=str(old),sha=sha(old)));reused+=1
        if path.exists():
            r=read(path)
            assert r['source_id']==item['key'] and np.allclose(r['context']['candidate_mw'],item['record']['context']['candidate_mw'],rtol=0,atol=1e-9)
        else:jobs.append((str(source),item,str(folder)))
        entries.append({k:v for k,v in item.items() if k not in ['record','input_record_file'] }|dict(path=str(path)))
    write(root/'inventory.json',entries)
    completed=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(_native,j):j[1]['id'] for j in jobs}
        for future in as_completed(futures):
            future.result();completed+=1
            write(root/'collection_progress.json',dict(new_completed=completed,new_required=len(jobs),reused_existing=len(items)-len(jobs),latest=futures[future]))
            print(f'native {completed}/{len(jobs)}: {futures[future]}',flush=True)
    hashes={e['path']:sha(e['path']) for e in entries}
    for path,h in dependencies.items():assert sha(path)==h
    for path,h in sources.items():assert sha(path)==h
    counts=dict(total=len(items),initial=sum(x['initial'] for x in items),candidate=sum(not x['initial'] for x in items),new_native_this_invocation=len(jobs),reused_this_invocation=len(items)-len(jobs))
    write(root/'collection_complete.json',dict(complete=True,counts=counts,records=hashes))
    return counts


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();print(collect(args.root,args.source,args.workers))
