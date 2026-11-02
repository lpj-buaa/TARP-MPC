"""Hash and protocol checks only. Never launches physics or fitting."""
from pathlib import Path
import argparse,json
import numpy as np
from tarp_boundary.io import read,sha,write
from tarp_acq_evidence.pipeline import verify_seal


def validate_reuse_provenance(old,new):
    newcode={str(Path(p).resolve()):h for p,h in new['code'].items()}
    newinputs={str(Path(p).resolve()):h for p,h in new['inputs'].items()}
    for path,h in old['code'].items():
        path=str(Path(path).resolve())
        if path not in newcode or newcode[path]!=h:raise ValueError('Historical physical code mismatch: '+path)
    for path,h in old['source_inputs'].items():
        path=str(Path(path).resolve())
        if path not in newinputs or newinputs[path]!=h:raise ValueError('Historical snapshot/source mismatch: '+path)


def verify_historical_reuse(root,inventory):
    from tarp_boundary.io import PROJECT
    old=PROJECT/'artifacts/boundary_v2/boundary_formal_s42_v2';new=read(root/'physical_protocol.json')
    validate_reuse_provenance(read(old/'collection_protocol.json'),new)
    for path,h in read(old/'measurement_protocol.json')['code'].items():
        if path in new['code'] and new['code'][path]!=h:raise ValueError('Historical MPC measurement code mismatch')
    for path,h in read(old/'collection_complete.json')['records'].items():assert sha(path)==h,path
    for path,h in read(old/'measurement_complete.json')['outputs'].items():assert sha(path)==h,path
    checked=0
    for c in inventory:
        prototype=c['prototype']
        for item in c['items']:
            folder=Path(item['path']).parent
            if not (folder/'reuse.json').exists():continue
            reuse=read(folder/'reuse.json');assert sha(item['path'])==reuse['sha']==sha(reuse['record'])
            record=read(item['path']);assert record['source_id']==prototype['source_id']
            assert record['pool_hour']==prototype['pool_hour'] and record['profile_id']==prototype['profile_id']
            assert record['context']['now']==prototype['context']['now']
            for name in ['preference_anchor','backup_energy_mwh']:
                np.testing.assert_allclose(record['context'][name],prototype['context'][name],rtol=0,atol=1e-10)
            np.testing.assert_allclose(record['context']['view']['ess_energy_mwh'],prototype['context']['view']['ess_energy_mwh'],rtol=0,atol=1e-10)
            checked+=1
    write(root/'historical_reuse_verification.json',dict(verified=checked,snapshot_provenance_verified=True,
        old_collection_protocol_sha=sha(old/'collection_protocol.json'),old_measurement_protocol_sha=sha(old/'measurement_protocol.json')))
    return checked


def verify(root):
    root=Path(root);counts={}
    if (root/'physical_protocol.json').exists():
        p=read(root/'physical_protocol.json')
        for file,h in {**p['inputs'],**p['code']}.items():
            if sha(file)!=h:raise ValueError('Frozen physical input changed: '+file)
        counts['frozen_physical_files']=len(p['inputs'])+len(p['code'])
    if (root/'inventory.json').exists():
        inventory=read(root/'inventory.json');complete=0
        for c in inventory:
            for item in c['items']:
                path=Path(item['path']);m=path.parent/'complete.json'
                if not m.exists():continue
                marker=read(m)
                assert sha(path)==marker['record_sha'] and sha(path.parent/'measurement.json')==marker['measurement_sha']
                complete+=1
        counts['sealed_branches']=complete
        if any((Path(i['path']).parent/'reuse.json').exists() for c in inventory for i in c['items']):
            counts['verified_historical_reuse']=verify_historical_reuse(root,inventory)
    for f in [root/'dataset_complete.json',*root.glob('response_s*/complete.json'),*root.glob('policies/*/complete.json'),root/'evaluation_complete.json']:
        if f.exists():verify_seal(f)
    if (root/'evaluation.json').exists():
        evaluation=read(root/'evaluation.json')
        for r in evaluation['rows']:
            assert len(r['queries'])==len(set(r['queries']))<=r['budget']
        counts['evaluated_episodes']=len(evaluation['rows'])
    for file in root.glob('response_s*/fit.json'):
        fit=read(file);assert set(fit['train_sources']).isdisjoint(fit['tune_sources']) or fit['smoke_resubstitution']
    for file in root.glob('policies/*/fit.json'):
        fit=read(file);assert set(fit['train_sources']).isdisjoint(fit['tune_sources']) or fit['smoke_resubstitution']
    counts['verified']=True;write(root/'verification.json',counts);return counts


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(verify(a.root),ensure_ascii=False))
