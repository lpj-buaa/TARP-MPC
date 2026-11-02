from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import numpy as np
from tarp_boundary.io import read,write,sha,bind,PROJECT,source_hashes
from tarp_boundary.learning import information_reward
from tarp_boundary.metrics import evaluate
from tarp_info.cache import Oracle
from .data import prepare,frozen_files,RESPONSE_GROUPS
from .experiment import experiment,ranking_metrics,from_response,VARIANTS


def seal(path,outputs):
    write(path,dict(outputs={str(Path(p).resolve()):sha(p) for p in outputs}))


def verify_seal(path):
    for p,h in read(path)['outputs'].items():
        if not Path(p).is_file() or sha(p)!=h:raise ValueError('Output hash mismatch: '+p)


def finish_rankings(bundle,result):
    for snap in result['rankings']:
        observed=bundle['initial']+result['queries'][:snap['budget']]
        lookup={p['id']:p for p in bundle['public']}
        rows=[dict(lookup[ident],**bundle['answers'][ident]) for ident in snap['candidate_ids']]
        truth=[information_reward(r,observed) for r in rows]
        snap['truth']=truth
        snap['sources']=ranking_metrics(snap['candidate_ids'],snap['scores'],truth,snap['keys'])
        proxy=[information_reward(from_response(r,p),observed) for r,p in zip(rows,snap['response_predictions'])]
        snap['derived_scores']=proxy.tolist() if isinstance(proxy,np.ndarray) else proxy
        snap['derived_sources']=ranking_metrics(snap['candidate_ids'],proxy,truth,snap['keys'])
        for name,(lo,hi) in RESPONSE_GROUPS.items():
            error=np.abs(np.array(snap['response_predictions'])[:,lo:hi]-np.array([r['response'] for r in rows])[:,lo:hi]).mean(1)
            snap.setdefault('response_errors',{})[name]={k:float(error[np.array(snap['keys'])==k].mean()) for k in sorted(set(snap['keys']))}
    result['evaluation']=evaluate(bundle,result)
    return result


def run(run_id,mode='formal',resume=False,until='complete'):
    source=PROJECT/'artifacts/boundary_v2/boundary_formal_s42_v2'
    root=PROJECT/'artifacts/acquisition_evidence_v3'/run_id
    if root.exists() and not resume:raise ValueError('RunId already exists; use --resume or a new RunId')
    root.mkdir(parents=True,exist_ok=True)
    cfg=dict(mode=mode,seeds=[42,43,44] if mode=='formal' else [42],budget=128 if mode=='formal' else 8,
             warm=100 if mode=='formal' else 3,updates=10 if mode=='formal' else 1,
             variants=VARIANTS,learning_rate=.0003,aux_weight=1,hidden=[128,128],exploration=.25,
             matched_history='frozen Random prefixes; exact same candidate IDs at each checkpoint',
             auxiliary='stationary 60 continuous outcomes, physical-state/action only',
             primary='matched-history source-mean Spearman at budget64; average seeds before source bootstrap',
             independent_units=6,new_physical_simulations=0,new_lp_solves=0,old_models_training=0)
    frozen=frozen_files(source,PROJECT);code=source_hashes(PROJECT/'src/tarp_acq_evidence')
    bind(root/'protocol.json',dict(config=cfg,source=frozen,code=code))
    if (root/'dataset_complete.json').exists():
        verify_seal(root/'dataset_complete.json');bundle=read(root/'bundle.json')
    else:
        bundle=prepare(source,root/'bundle.json');seal(root/'dataset_complete.json',[root/'bundle.json'])
    executed=[];skipped=[]
    if until!='prepare':
        for protocol in ['matched','active']:
            for seed in cfg['seeds']:
                schedule=[r['id'] for r in read(source/f'acquisition/random_s{seed}.json')['queries']]
                for variant in VARIANTS+['local_interpolation']:
                    label=f'{protocol}/{variant}_s{seed}';out=root/'experiments'/label
                    complete=out/'complete.json'
                    if complete.exists():verify_seal(complete);skipped.append(label);continue
                    print('starting '+label,flush=True)
                    result=experiment(bundle['public'],bundle['initial'],Oracle(bundle['answers']),variant,cfg['budget'],seed,
                                      fixed=schedule if protocol=='matched' else None,warm=cfg['warm'],updates=cfg['updates'],
                                      checkpoints=[16,32,64,128] if mode=='formal' else [4,8],checkpoint_dir=out/'checkpoints')
                    finish_rankings(bundle,result);write(out/'result.json',result)
                    seal(complete,[out/'result.json',*sorted((out/'checkpoints').glob('*.pt'))])
                    executed.append(label)
                    print('completed '+label,flush=True)
        baseline={}
        for seed in cfg['seeds']:
            for method,folder in [('random','acquisition'),('bracket_random','acquisition'),('parameter_sobol','baselines')]:
                result=read(source/f'{folder}/{method}_s{seed}.json')
                baseline[f'{method}_s{seed}']=evaluate(bundle,result)
        write(root/'baselines.json',baseline)
    for p,h in frozen.items():
        if sha(p)!=h:raise ValueError('Frozen input hash changed: '+p)
    for p,h in code.items():
        if sha(p)!=h:raise ValueError('Run code hash changed: '+p)
    outputs=list((root/'experiments').glob('*/*/result.json'))
    # Glob deliberately follows the two-level protocol/method directory structure.
    cumulative=sum(read(p)['acquisition_training_calls'] for p in outputs)
    final=dict(run_id=run_id,complete=until!='prepare',executed=executed,skipped=skipped,
               acquisition_training_calls=cumulative,executed_training_calls=sum('local_interpolation' not in x for x in executed),
               native_simulations=0,mpc_solves=0,old_rl_training_calls=0,forecaster_training_calls=0,expert_training_calls=0,
               frozen_files=len(frozen),frozen_hashes_verified=True,mode=mode,time_utc=datetime.now(timezone.utc).isoformat())
    write(root/'status.json',final)
    return final


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--mode',choices=['formal','smoke'],default='formal')
    p.add_argument('--resume',action='store_true');p.add_argument('--until',choices=['prepare','complete'],default='complete')
    args=p.parse_args();print(json.dumps(run(args.run_id,args.mode,args.resume,args.until),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
