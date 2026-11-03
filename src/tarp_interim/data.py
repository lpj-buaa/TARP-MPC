from copy import deepcopy
from pathlib import Path
from collections import Counter
from tarp_boundary.io import read,write,sha,bind
from tarp_acq_evidence.pipeline import seal,verify_seal

HOLDOUT={'2024-04-01_s102','2024-08-19_s112'}


def select_contexts(contexts,completion_path,holdout):
    selected=[]
    for c in contexts:
        if c['prototype']['role']=='confirm':continue
        path=Path(completion_path(c))
        if not path.exists():continue
        marker=read(path)
        if not marker.get('complete') or marker['items']!=len(c['items']):raise ValueError('Incomplete context seal')
        for p,h in marker['outputs'].items():
            if sha(p)!=h:raise ValueError('Context seal changed: '+p)
        for i in c['items']:
            folder=Path(i['path']).parent;m=read(folder/'complete.json')
            if sha(i['path'])!=m['record_sha'] or sha(folder/'measurement.json')!=m['measurement_sha']:
                raise ValueError('Branch seal changed: '+i['id'])
        own=deepcopy(c);original=c['prototype']['role'];role='development_test' if c['prototype']['source_id'] in holdout else original
        own['prototype'].update(original_role=original,role=role)
        for i in own['items']:i.update(original_role=original,role=role)
        selected.append(own)
    return selected


def snapshot(source,out):
    source,out=Path(source),Path(out);out.mkdir(parents=True,exist_ok=True)
    marker=out/'dataset_complete.json'
    if marker.exists():verify_seal(marker);return read(out/'dataset.json')
    inventory=read(source/'inventory.json')
    contexts=select_contexts(inventory,lambda c:source/'contexts'/c['key']/'complete.json',HOLDOUT)
    assert all(len(c['items'])==32 for c in contexts)
    if {c['prototype']['source_id'] for c in contexts if c['prototype']['role']=='development_test'}!=HOLDOUT:
        raise ValueError('Both predetermined development holdouts must be complete')
    sources={role:sorted({c['prototype']['source_id'] for c in contexts if c['prototype']['role']==role}) for role in {c['prototype']['role'] for c in contexts}}
    train=set(sources.get('train',[])+sources.get('meta_train',[]));tune=set(sources.get('tune',[])+sources.get('meta_tune',[]))
    assert train.isdisjoint(tune|HOLDOUT) and tune.isdisjoint(HOLDOUT)
    source_files={str(source/'inventory.json'):sha(source/'inventory.json'),str(source/'physical_protocol.json'):sha(source/'physical_protocol.json')}
    for c in contexts:
        complete=source/'contexts'/c['key']/'complete.json';source_files[str(complete)]=sha(complete)
        for i in c['items']:
            f=Path(i['path']).parent
            for p in [f/'record.json',f/'measurement.json',f/'complete.json']:source_files[str(p)]=sha(p)
    bind(out/'snapshot_protocol.json',dict(source=str(source.resolve()),source_files=source_files,
        holdout=sorted(HOLDOUT),original_confirm_used=0,complete_contexts=len(contexts),branches=sum(len(c['items']) for c in contexts),sources=sources))
    write(out/'inventory.json',contexts)
    from tarp_sequence.data import pack
    data=pack(out)
    data.update(scope='interim development holdouts from original training sources; original formal confirm unused',
                original_confirm_used=0,development_holdout=sorted(HOLDOUT),original_role_mapping={c['key']:c['prototype']['original_role'] for c in contexts})
    write(out/'dataset.json',data)
    seal(marker,[out/'dataset.json',out/'inventory.json',out/'snapshot_protocol.json'])
    return data
