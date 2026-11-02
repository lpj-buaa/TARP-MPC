import argparse
from pathlib import Path
from .io import PROJECT,read,write,sha,bind,source_hashes


def fit(root,seeds=(42,43,44),budget=128):
    from .learning import acquire
    from .metrics import evaluate
    from tarp_info.cache import Oracle
    bundle=read(root/'bundle.json')
    code=source_hashes(PROJECT/'src/tarp_boundary',PROJECT/'src/tarp_info',PROJECT/'src/tarp37')
    protocol=dict(bundle=sha(root/'bundle.json'),code=code,budget=budget,seeds=list(seeds),
        reward_weights=[1/3]*3,novelty_scale=.1,adjacent_dose=1/13,turn_tolerance=.01,warm_updates=100,updates_per_query=10,
        primary_budget=64,forecaster_training=0,old_models_training=0,
        knowledge_state='18 public/revealed-evidence features; identical inputs in all learned arms; self excluded during relabel')
    bind(root/'acquisition_protocol.json',protocol)
    methods=['random','sobol','error_td3','boundary_reward_only','bracket_random','boundary_td3'];summaries=[];executed=0
    for seed in seeds:
        for method in methods:
            path=root/'acquisition'/f'{method}_s{seed}.json'
            if path.exists():
                result=read(path);assert result['protocol']==protocol
                if result.get('checkpoint'):assert sha(result['checkpoint'])==result['checkpoint_hash']
                print(method,seed,'verified reused',flush=True)
            else:
                checkpoint=root/'acquisition'/f'{method}_s{seed}.pt'
                result=acquire(bundle['public'],bundle['initial'],Oracle(bundle['answers']),method,budget,seed,checkpoint_path=checkpoint)
                if result['acquisition_training_calls']:result.update(checkpoint=str(checkpoint),checkpoint_hash=sha(checkpoint))
                result['protocol']=protocol;result['evaluation']=evaluate(bundle,result);write(path,result)
                executed+=result['acquisition_training_calls']
                row=next(x for x in result['evaluation'] if x['budget']==min(64,budget))
                print(method,seed,'budget',row['budget'],'boundary',row['boundary_pairs_found'],'/',row['boundary_pairs_total'],'distance',row['distance'],flush=True)
            summaries.append(dict(method=method,seed=seed,evaluation=result['evaluation']))
    for file,h in code.items():assert sha(file)==h
    outputs={str(f):sha(f) for f in (root/'acquisition').glob('*') if f.suffix in ['.json','.pt']}
    write(root/'acquisition_complete.json',dict(complete=True,arms=summaries,outputs=outputs,executed_acquisition_training_calls=executed,
        cumulative_acquisition_training_calls=3*len(seeds),forecaster_training_calls=0,old_models_training_calls=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['bundle','fit']);p.add_argument('--run-id',required=True);p.add_argument('--source',default=str(PROJECT.parent/'tarp_37_risk_audit'))
    args=p.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):raise ValueError('Invalid RunId')
    root=PROJECT/'artifacts/boundary_v2'/args.run_id
    if args.stage=='bundle':
        from .measurements import make_bundle
        inputs=[root/'teacher.json',root/'inventory.json',Path(__file__),PROJECT/'src/tarp_boundary/measurements.py',PROJECT/'src/tarp_info/profiles.py']
        for item in read(root/'inventory.json'):inputs.extend([Path(item['path']),root/'measurements'/(item['id']+'.json')])
        bind(root/'bundle_protocol.json',dict(inputs={str(f.resolve()):sha(f) for f in inputs}))
        if (root/'bundle.json').exists():print('bundle preserved');return
        bundle=make_bundle(root,args.source);print('bundle',len(bundle['initial']),len(bundle['public']))
    else:fit(root)


if __name__=='__main__':main()
