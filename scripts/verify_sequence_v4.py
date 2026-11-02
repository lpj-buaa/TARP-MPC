"""Hash and protocol checks only. Never launches physics or fitting."""
from pathlib import Path
import argparse,json
from tarp_boundary.io import read,sha,write
from tarp_acq_evidence.pipeline import verify_seal


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
