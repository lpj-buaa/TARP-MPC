import argparse
import json
from pathlib import Path
from .assets import PROJECT
from .collection import read
from .learning import learning_task

TASKS=['pack_initial','fit_initial','explore','pack_augmented','fit_experts','predict','fit_fusion','calibrate','evaluate']


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-run',required=True); p.add_argument('--run-id',required=True)
    p.add_argument('--mode',choices=['pilot','formal'],default='pilot'); p.add_argument('--task',choices=TASKS+['all'],default='all')
    p.add_argument('--device',default='auto'); p.add_argument('--resume',action='store_true'); p.add_argument('--skip-rl',action='store_true')
    p.add_argument('--policy',choices=['td3','sobol','random'],default='td3'); p.add_argument('--queries',type=int)
    p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    config=read(PROJECT/f'configs/training_{a.mode}.json'); config['seed']=a.seed
    config['acquisition_policy']='none' if a.skip_rl else a.policy
    config['acquisition_queries']=0 if a.skip_rl else (a.queries or config['query_episodes'])
    tasks=TASKS if a.task=='all' else [a.task]
    for task in tasks:
        if a.skip_rl and task in ('explore','pack_augmented'): continue
        if task=='explore':
            from .acquisition import acquire
            acquire(a.data_run,a.run_id,config,a.device,a.policy,a.queries or config['query_episodes'],a.resume)
        else:
            query_folder=PROJECT/'artifacts/retraining'/a.run_id/'explore/queries' if task=='pack_augmented' else None
            learning_task(a.data_run,a.run_id,task,config,a.device,a.resume,query_folder)
        print(task+': complete',flush=True)
    print(json.dumps(dict(run_id=a.run_id,root=str(PROJECT/'artifacts/retraining'/a.run_id),tasks=tasks),indent=2))


if __name__=='__main__': main()
