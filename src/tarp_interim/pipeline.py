from pathlib import Path
import argparse,time
import numpy as np
from tarp_boundary.io import PROJECT,read,write,sha,bind
from tarp_acq_evidence.pipeline import seal,verify_seal
from tarp_sequence.data import load_cases
from .data import snapshot


def run(run_id,device='cuda',resume=False,stage='all'):
    root=PROJECT/'artifacts/sequence_interim'/run_id
    if root.exists() and not resume:raise ValueError('Run exists; use --resume')
    root.mkdir(parents=True,exist_ok=True)
    source=PROJECT/'artifacts/sequence_v4/sequence_formal_s42_v1'
    bind(root/'run_protocol.json',dict(source=str(source),run_id=run_id,seeds=[42,43,44],scope='interim development; original confirm unused'))
    executed=[];skipped=[];start=time.perf_counter()
    if not (root/'dataset_complete.json').exists():snapshot(source,root);executed.append('snapshot')
    else:verify_seal(root/'dataset_complete.json');skipped.append('snapshot')
    if stage=='prepare':
        print(read(root/'dataset.json')['counts'],flush=True);return
    cases=load_cases(root);seeds=[42,43,44]
    code={str(p.resolve()):sha(p) for name in ['tarp_sequence','tarp_interim'] for p in (PROJECT/'src'/name).glob('*.py')}
    bind(root/'learning_protocol.json',dict(code=code,dataset_sha=sha(root/'dataset.json'),seeds=seeds,device=device,
        response_epochs=100,patience=12,batch_size=128,policy_batches=120,episodes_per_batch=8,policy_update_epochs=4,
        budgets=[8,16,24],learning_rate=.0003,monitoring='no gradient/no random draws',original_confirm_used=0))
    if stage in ['all','fit']:
        from .training import fit_response
        from tarp_sequence.learning import fit_policy
        for seed in seeds:
            target=root/f'response_s{seed}';marker=target/'complete.json'
            if marker.exists():verify_seal(marker);skipped.append(target.name)
            else:
                print(f'fit response {seed}',flush=True);fit_response(cases,target,seed,device)
                seal(marker,[target/'model.pt',target/'last_model.pt',target/'fit.json',target/'predictions.npz']);executed.append(target.name)
            with np.load(target/'predictions.npz') as saved:predictions={k:saved[k] for k in saved.files}
            for method in ['ppo','myopic','group']:
                target=root/f'policies/{method}_s{seed}';marker=target/'complete.json'
                if marker.exists():verify_seal(marker);skipped.append(target.name)
                else:
                    print(f'fit policy {method} {seed}',flush=True);fit_policy(cases,predictions,target,method,seed,device)
                    seal(marker,[target/'model.pt',target/'fit.json']);executed.append(target.name)
    if stage in ['all','evaluate']:
        from tarp_sequence.learning import evaluate
        dependencies={}
        for seed in seeds:
            response=root/f'response_s{seed}';verify_seal(response/'complete.json');dependencies[str(response/'predictions.npz')]=sha(response/'predictions.npz')
            for method in ['ppo','myopic','group']:
                target=root/f'policies/{method}_s{seed}';verify_seal(target/'complete.json');dependencies[str(target/'model.pt')]=sha(target/'model.pt')
        bind(root/'evaluation_protocol.json',dict(dataset_sha=sha(root/'dataset.json'),dependencies=dependencies))
        marker=root/'evaluation_complete.json'
        if marker.exists():verify_seal(marker);skipped.append('evaluate')
        else:
            print('evaluate eight methods',flush=True);evaluate(root,cases,seeds,device)
            seal(marker,[root/'evaluation.json']);executed.append('evaluate')
        marker=root/'diagnostics_complete.json'
        if marker.exists():verify_seal(marker);skipped.append('diagnose')
        else:
            from .analysis import diagnose
            print('diagnose outcomes and marginal ranking',flush=True);diagnose(root)
            seal(marker,[root/'diagnostics.json']);executed.append('diagnose')
    for p,h in code.items():
        if sha(p)!=h:raise ValueError('Bound code changed: '+p)
    for p,h in read(root/'snapshot_protocol.json')['source_files'].items():
        if sha(p)!=h:raise ValueError('Frozen source changed: '+p)
    for marker in [*root.glob('response_s*/complete.json'),*root.glob('policies/*/complete.json'),root/'evaluation_complete.json',root/'diagnostics_complete.json']:
        if marker.exists():verify_seal(marker)
    status=dict(executed=executed,skipped=skipped,new_response_fits=sum(x.startswith('response_s') for x in executed),
                new_policy_fits=sum(x.startswith(('ppo_s','myopic_s','group_s')) for x in executed),
                new_simulations=0,old_rl_training=0,old_forecaster_training=0,seconds=time.perf_counter()-start)
    write(root/'invocation_status.json',status);print(status,flush=True);return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='sequence_interim_s42_v1');p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true');p.add_argument('--stage',choices=['prepare','all','fit','evaluate','verify'],default='all')
    a=p.parse_args();run(a.run_id,a.device,a.resume,a.stage)
