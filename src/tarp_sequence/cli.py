from pathlib import Path
import argparse,json,time
from tarp_boundary.io import read,write,sha,bind,PROJECT
from tarp_acq_evidence.pipeline import seal,verify_seal


def run(run_id,stage='fit',mode='formal',device='cpu'):
    root=PROJECT/'artifacts/sequence_v4'/run_id
    if stage=='pack':
        from .data import pack
        if (root/'dataset_complete.json').exists():verify_seal(root/'dataset_complete.json');return dict(skipped=['pack'])
        if not (root/'collection_complete.json').exists():raise ValueError('Physical collection is not complete')
        pack(root);seal(root/'dataset_complete.json',[root/'dataset.json']);return dict(executed=['pack'])
    from .data import load_cases
    import torch
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(1);cases=load_cases(root);seeds=[42] if mode=='smoke' else [42,43,44]
    code={str(p.resolve()):sha(p) for p in (PROJECT/'src/tarp_sequence').glob('*.py') if p.name!='corpus.py'}
    protocol=dict(dataset_sha=sha(root/'dataset.json'),code=code,seeds=seeds,mode=mode,
                  budgets=[8,16,24],batches=3 if mode=='smoke' else 120,episodes_per_batch=8,policy_epochs=4,
                  gamma={'ppo':1.,'myopic':0.,'group':1.},group_size=4,forecaster_training=0,old_rl_training=0)
    bind(root/'learning_protocol.json',protocol);executed=[];skipped=[]
    if stage in ['fit','all']:
        from .response import fit
        from .learning import fit_policy
        import numpy as np
        for seed in seeds:
            out=root/f'response_s{seed}';marker=out/'complete.json'
            bind(out/'inputs.json',dict(dataset_sha=sha(root/'dataset.json'),seed=seed))
            if marker.exists():verify_seal(marker);skipped.append(out.name)
            else:
                print('fit response',seed,flush=True);fit(cases,out,seed,device,mode=='smoke')
                seal(marker,[out/'inputs.json',out/'model.pt',out/'fit.json',out/'predictions.npz']);executed.append(out.name)
            with np.load(out/'predictions.npz') as f:predictions={k:f[k] for k in f.files}
            for method in ['ppo','myopic','group']:
                target=root/f'policies/{method}_s{seed}';marker=target/'complete.json'
                bind(target/'inputs.json',dict(dataset_sha=sha(root/'dataset.json'),response_predictions_sha=sha(out/'predictions.npz'),method=method,seed=seed))
                if marker.exists():verify_seal(marker);skipped.append(target.name)
                else:
                    print('fit policy',method,seed,flush=True);fit_policy(cases,predictions,target,method,seed,device,mode=='smoke')
                    seal(marker,[target/'inputs.json',target/'model.pt',target/'fit.json']);executed.append(target.name)
    if stage in ['evaluate','all']:
        from .learning import evaluate
        dependencies={}
        for seed in seeds:
            response=root/f'response_s{seed}';verify_seal(response/'complete.json')
            dependencies[str(response/'predictions.npz')]=sha(response/'predictions.npz')
            for method in ['ppo','myopic','group']:
                modeldir=root/f'policies/{method}_s{seed}';verify_seal(modeldir/'complete.json')
                dependencies[str(modeldir/'model.pt')]=sha(modeldir/'model.pt')
        bind(root/'evaluation_protocol.json',dict(dataset_sha=sha(root/'dataset.json'),model_inputs=dependencies))
        marker=root/'evaluation_complete.json'
        if marker.exists():verify_seal(marker);skipped.append('evaluate')
        else:
            evaluate(root,cases,seeds,device,mode=='smoke');seal(marker,[root/'evaluation.json']);executed.append('evaluate')
    for p,h in code.items():assert sha(p)==h,p
    status=dict(stage=stage,executed=executed,skipped=skipped,device=device,
                new_response_fits_this_call=sum(x.startswith('response_') for x in executed),
                new_policy_fits_this_call=sum(x.startswith(('ppo_','myopic_','group_')) for x in executed),old_models_training=0)
    write(root/('learning_status.json' if stage!='evaluate' else 'evaluation_status.json'),status);return status


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=['pack','fit','evaluate','all'],default='all')
    p.add_argument('--mode',choices=['formal','smoke'],default='formal');p.add_argument('--device',default='cpu')
    a=p.parse_args();print(json.dumps(run(a.run_id,a.stage,a.mode,a.device),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
