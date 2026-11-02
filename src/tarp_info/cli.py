import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .cache import prepare,load_json,Oracle
from .acquire import run_arm


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evaluate(bundle,result):
    # Only the post-query evaluator receives all hidden answers.
    public={p['id']:p for p in bundle['public']};answers=bundle['answers'];keys=sorted(set(p['key'] for p in bundle['public']))
    checkpoints=[0,*[x for x in [8,16,32,64,128] if x<=result['budget']]]
    if result['budget'] not in checkpoints:checkpoints.append(result['budget'])
    records=[]
    for budget in checkpoints:
        selected=result['queries'][:budget];per=[]
        for key in keys:
            pool=[np.array(answers[id]['profile']) for id,p in public.items() if p['key']==key and answers[id]['profile'] is not None]
            known=[np.array(x['profile']) for x in bundle['initial']+selected if x['key']==key and x['profile'] is not None]
            if not pool or not known:continue
            distances=np.linalg.norm(np.array(pool)[:,None,:]-np.array(known)[None,:,:],axis=-1).min(1)
            per.append(dict(key=key,source_id=key.split('|')[0],radius=float(distances.max()),mean_distance=float(distances.mean()),
                coverage_005=float((distances<=.05).mean()),coverage_010=float((distances<=.1).mean()),coverage_020=float((distances<=.2).mean())))
        query_joint=sum(all(q['known'][2:]) and all(q['event'][2:]) for q in selected)
        records.append(dict(budget=budget,contexts=per,radius_macro=float(np.mean([r['radius'] for r in per])),
            distance_macro=float(np.mean([r['mean_distance'] for r in per])),coverage_010=float(np.mean([r['coverage_010'] for r in per])),
            joint=int(query_joint),induced_joint=sum(q['induced_joint'] for q in selected),late_service=sum(q['late_service'] for q in selected),
            unknown_profiles=sum(q['profile'] is None for q in selected),
            redundant_005=sum(q['insertion_novelty'] is not None and q['insertion_novelty']<=.05 for q in selected)))
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','replay']);p.add_argument('--source',required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--budget',type=int,default=128);p.add_argument('--seeds',default='42,43,44');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not a.run_id or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in a.run_id):raise ValueError('Invalid RunId')
    project=Path(__file__).resolve().parents[2];root=project/'artifacts/information'/a.run_id;root.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':
        if (root/'cache.json').exists():raise ValueError('Cache exists; preserve it and use another RunId')
        bundle=prepare(a.source,root,root/'utility.json')
        print(json.dumps(bundle['manifest'],indent=2));return
    bundle=load_json(root/'cache.json')
    code={str(f):digest(f) for f in Path(__file__).parent.glob('*.py')}
    protocol=dict(cache_hash=digest(root/'cache.json'),code=code,budget=a.budget,seeds=a.seeds,
        fixed_metrics=['mean_distance','radius','coverage_005','coverage_010','coverage_020'],warm=100,updates=10,exploration=.25)
    path=root/'protocol.json'
    if path.exists() and load_json(path)!=protocol:raise ValueError('Protocol changed; use a new RunId')
    path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    summary=[]
    for seed in map(int,a.seeds.split(',')):
        for method in ['random','sobol','error_td3','space_td3']:
            out=root/f'{method}_s{seed}.json'
            if out.exists():
                if not a.resume:raise ValueError('Arm exists; use --resume')
                result=load_json(out);assert result['cache_hash']==protocol['cache_hash'] and result['code']==code
            else:
                print(f'{method} seed{seed}: {a.budget} cached simulator queries',flush=True)
                result=run_arm(bundle['public'],bundle['initial'],Oracle(bundle['answers']),method,a.budget,seed)
                result.update(cache_hash=protocol['cache_hash'],code=code)
                result['evaluation']=evaluate(bundle,result)
                out.write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
            summary.append(dict(method=method,seed=seed,**result['evaluation'][-1]))
            print(method,seed,'distance',round(result['evaluation'][-1]['distance_macro'],5),'joint',result['evaluation'][-1]['joint'],flush=True)
    (root/'summary.json').write_text(json.dumps(dict(complete=True,arms=summary,scope=bundle['manifest']['scope'],risk_forecaster_training_calls=0),indent=2),encoding='utf-8')


if __name__=='__main__':main()
