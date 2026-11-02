"""Read-only validation before any Resume can rewrite completion manifests."""
import argparse,json,hashlib
from pathlib import Path


def context_difference(a,b):
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():return float('inf')
        return max((context_difference(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b):return float('inf')
        return max((context_difference(x,y) for x,y in zip(a,b)),default=0.)
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return abs(a-b)
    return 0. if a==b else float('inf')


def same_context(a,b):return context_difference(a,b)<=1e-10


def verify(root):
    root=Path(root);count=0
    for name,field in [('collection_complete.json','records'),('measurement_complete.json','outputs'),('acquisition_complete.json','outputs')]:
        marker=root/name
        if not marker.exists():continue
        saved=json.loads(marker.read_text(encoding='utf-8'))
        for file,digest in saved[field].items():
            actual=hashlib.sha256(Path(file).read_bytes()).hexdigest()
            if actual!=digest:raise ValueError('Published artifact changed: '+file)
            count+=1
    print('Published immutable outputs verified:',count)


def final_verify(root):
    import numpy as np
    from collections import Counter
    root=Path(root);project=Path(__file__).resolve().parents[1]
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    verify(root)
    for name,fields in [('collection_protocol.json',['code','source_inputs']),('measurement_protocol.json',['code']),
                        ('teacher_protocol.json',['inputs']),('bundle_protocol.json',['inputs']),('acquisition_protocol.json',['code'])]:
        protocol=read(root/name)
        for field in fields:
            for f,h in protocol[field].items():assert sha(f)==h,f
    bundle=read(root/'bundle.json');assert len(bundle['initial'])==18 and len(bundle['public'])==144
    assert all(set(p)==set(['id','key','source_id','family','dose','state','action']) for p in bundle['public'])
    protocol=read(root/'acquisition_protocol.json');assert protocol['bundle']==sha(root/'bundle.json')
    training=0;sets={}
    for f in sorted((root/'acquisition').glob('*.json')):
        r=read(f);assert r['protocol']==protocol and r['budget']==128
        assert len(set(r['oracle_calls']))==128 and r['oracle_calls']==[q['id'] for q in r['queries']]
        assert set(Counter(q['key'] for q in r['queries']).values())=={21,22}
        assert r['old_models_training_calls']==r['risk_forecaster_training_calls']==0
        if r.get('checkpoint'):assert sha(r['checkpoint'])==r['checkpoint_hash']
        training+=r['acquisition_training_calls'];sets[f.stem]=[q['id'] for q in r['queries']]
    assert len(sets)==18 and training==9
    items=read(root/'inventory.json');assert len(items)==162
    contexts={};hardware=[];known_hours=0;max_initial_difference=0.
    for item in items:
        r=read(item['path']);ctx={k:v for k,v in r['context'].items() if k!='candidate_mw'}
        if item['key'] in contexts:
            # Inputs are identical before applying the branch intervention.
            difference=context_difference(ctx,contexts[item['key']]);max_initial_difference=max(max_initial_difference,difference)
            assert same_context(ctx,contexts[item['key']]),(item['id'],difference)
        else:contexts[item['key']]=ctx
        hardware.append(r['hardware'])
        for row in r['rows']:
            if not row.get('physical_valid'):continue
            assert row['max_bus_residual_mva']<1e-5
            assert min(row['voltage'])>=.93-1e-6 and max(row['voltage'])<=1.07+1e-6
            assert max(row['loading_ratio'])<=1+1e-6 and row['generator_bound_excess']<=1e-6
            assert row['battery_balance_error_kwh']<1e-5 and row['ess_balance_error_mwh']<1e-7
            known_hours+=1
        m=read(root/'measurements'/(item['id']+'.json'))
        for value in m['measurements']:
            if value['known']:assert all(a['status']==0 or (a['task']=='primary' and a['status']==2) for a in value['attempts'])
    assert len(contexts)==6 and all(x==hardware[0] for x in hardware)
    assert hardware[0]['line_limit_ka']==.9 and hardware[0]['voltage_limits_pu']==[.93,1.07]
    import subprocess
    assert not subprocess.check_output(['git','diff','e604c1b','--name-only','--','src/tarp37','src/tarp_info'],cwd=project,text=True).strip()
    for folder in ['tarp_37_risk_audit','tarp_37_voltage_v5']:
        assert not subprocess.check_output(['git','-C',str(project.parent/folder),'status','--porcelain'],text=True).strip()
    result=dict(complete=True,initial_rows=18,candidate_rows=144,sources=6,arms=18,query_budget_each=128,
        new_acquisition_training_calls=training,old_models_training_calls=0,forecaster_training_calls=0,
        native_branches=162,known_physical_hours=known_hours,same_initial_contexts_verified=True,
        maximum_initial_numeric_difference=max_initial_difference,initial_comparison_atol=1e-10,
        frozen_physics_verified=True,published_output_hashes_verified=True)
    (root/'final_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--final',action='store_true');a=p.parse_args()
    final_verify(a.root) if a.final else verify(a.root)
