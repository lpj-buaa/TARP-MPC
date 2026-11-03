from pathlib import Path
import argparse,shutil,time
import numpy as np
import torch
from tarp_boundary.io import PROJECT,read,write,sha,bind
from tarp_acq_evidence.pipeline import seal,verify_seal
from tarp_sequence.data import load_cases
from tarp_sequence.policy import SetPolicy
from .policy import ARMS,fit_policy,evaluate_policy
from .training import fit_conditional


def prepare(source,root,mode,device):
    source,root=Path(source),Path(root)
    verify_seal(source/'dataset_complete.json');verify_seal(source/'evaluation_complete.json')
    seeds=[42] if mode=='smoke' else [42,43,44]
    dependencies={str(p.resolve()):sha(p) for p in [source/'dataset.json',source/'evaluation.json']}
    for seed in seeds:
        for part in [source/f'response_s{seed}',source/f'policies/ppo_s{seed}']:
            verify_seal(part/'complete.json')
            for p in part.glob('*'):
                if p.is_file():dependencies[str(p.resolve())]=sha(p)
    code={str(p.resolve()):sha(p) for name in ['tarp_sequence','tarp_interim'] for p in (PROJECT/'src'/name).glob('*.py')}
    code.update({str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('*.py') if p.name not in ['analysis.py','report.py']})
    root.mkdir(parents=True,exist_ok=True)
    bind(root/'run_protocol.json',dict(source=str(source.resolve()),dependencies=dependencies,code=code,
        mode=mode,device=device,seeds=seeds,budgets=[8,16,24],response_epochs=2 if mode=='smoke' else 100,
        policy_batches=2 if mode=='smoke' else 120,rank_weight=.2,conditional_weight=.5,
        original_confirm_used=0,new_simulations=0,scope='Fixed development sources; not independent formal confirm',
        baseline_reused=True,response_parameters=136639,policy_parameters=121794))
    if (root/'dataset.json').exists():
        if sha(root/'dataset.json')!=sha(source/'dataset.json'):raise ValueError('Frozen dataset changed')
    else:shutil.copyfile(source/'dataset.json',root/'dataset.json')
    roles={}
    for c in load_cases(root):
        old=roles.setdefault(c['source_id'],c['role'])
        if old!=c['role']:raise ValueError('Source appears in multiple roles')
        if c['role']=='confirm':raise ValueError('Original confirm cannot enter development reuse')
    return read(root/'run_protocol.json')


def verify(root):
    root=Path(root);protocol=read(root/'run_protocol.json')
    for p,h in {**protocol['dependencies'],**protocol['code']}.items():
        if sha(p)!=h:raise ValueError('Frozen dependency changed: '+p)
    if sha(root/'dataset.json')!=sha(Path(protocol['source'])/'dataset.json'):raise ValueError('Frozen dataset changed')
    if (root/'analysis_protocol.json').exists():
        analysis=read(root/'analysis_protocol.json')
        for p,h in analysis['code'].items():
            if sha(p)!=h:raise ValueError('Frozen analysis code changed: '+p)
        if sha(root/'evaluation.json')!=analysis['evaluation_sha']:raise ValueError('Frozen evaluation changed for analysis')
    for p in [*root.glob('response_s*/complete.json'),*root.glob('policies/*/complete.json'),root/'evaluation_complete.json',root/'analysis_complete.json']:
        if p.exists():verify_seal(p)


def run(run_id,mode='reuse',device='cuda',resume=False,stage='all'):
    root=PROJECT/'artifacts/sequence_alignment'/run_id
    if root.exists() and not resume:raise ValueError('Run exists; use --resume or a new RunId')
    source=PROJECT/'artifacts/sequence_interim/sequence_interim_s42_v1'
    protocol=prepare(source,root,mode,device);cases=load_cases(root)
    executed=[];skipped=[];start=time.perf_counter()
    if stage in ['all','fit']:
        for seed in protocol['seeds']:
            target=root/f'response_s{seed}';marker=target/'complete.json'
            if marker.exists():verify_seal(marker);skipped.append(target.name)
            else:
                fit_conditional(cases,target,seed,device,protocol['response_epochs'])
                seal(marker,[target/f for f in ['model.pt','last_model.pt','fit.json','predictions.npz']]);executed.append(target.name)
            with np.load(source/f'response_s{seed}/predictions.npz') as f:base={k:f[k] for k in f.files}
            with np.load(target/'predictions.npz') as f:conditional={k:f[k] for k in f.files}
            for arm,(_,_,trained) in ARMS.items():
                target=root/f'policies/{arm}_s{seed}';marker=target/'complete.json'
                if marker.exists():verify_seal(marker);skipped.append(f'{arm}_s{seed}')
                else:
                    fit_policy(cases,conditional if trained else base,target,arm,seed,device,protocol['policy_batches'])
                    seal(marker,[target/'model.pt',target/'fit.json']);executed.append(f'{arm}_s{seed}')
    if stage in ['all','evaluate']:
        marker=root/'evaluation_complete.json'
        if marker.exists():verify_seal(marker);skipped.append('evaluation')
        else:
            rows=[r for r in read(source/'evaluation.json')['rows'] if r['seed'] in protocol['seeds'] and r['method'] in ['ppo','local','bracket_random','random','sobol']]
            evaluable=[c for c in cases if c['role'] not in {'train','meta_train'} and c['valid'].any()]
            for seed in protocol['seeds']:
                with np.load(source/f'response_s{seed}/predictions.npz') as f:base={k:f[k] for k in f.files}
                verify_seal(root/f'response_s{seed}/complete.json')
                with np.load(root/f'response_s{seed}/predictions.npz') as f:conditional={k:f[k] for k in f.files}
                for arm,(_,correction,trained) in ARMS.items():
                    target=root/f'policies/{arm}_s{seed}';verify_seal(target/'complete.json')
                    model=SetPolicy().to(device);model.load_state_dict(torch.load(target/'model.pt',map_location=device,weights_only=False)['model']);model.eval()
                    for budget in protocol['budgets']:
                        result=evaluate_policy(model,evaluable,conditional if trained else base,budget,device,correction)
                        rows.extend(dict(r,method=arm,seed=seed) for r in result)
            write(root/'evaluation.json',dict(rows=rows,baseline_rows_reused=True,new_simulations=0))
            seal(marker,[root/'evaluation.json']);executed.append('evaluation')
    if stage in ['all','analyze']:
        from .analysis import analyze
        marker=root/'analysis_complete.json'
        if marker.exists():verify_seal(marker);skipped.append('analysis')
        else:
            bind(root/'analysis_protocol.json',dict(code={str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('*.py') if p.name!='report.py'},evaluation_sha=sha(root/'evaluation.json')))
            analyze(root,device)
            seal(marker,[root/'analysis.json',root/'diagnostics.json']);executed.append('analysis')
    verify(root)
    result=dict(run_id=run_id,executed=executed,skipped=skipped,new_response_fits=sum(x.startswith('response_s') for x in executed),
        new_policy_fits=sum(x.startswith(tuple(ARMS)) for x in executed),new_simulations=0,old_rl_training=0,
        old_forecaster_training=0,seconds=time.perf_counter()-start)
    write(root/'invocation_status.json',result);print(result,flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='align_reuse_s42_v1');p.add_argument('--mode',choices=['smoke','reuse'],default='reuse')
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    p.add_argument('--stage',choices=['all','prepare','fit','evaluate','analyze','verify'],default='all')
    a=p.parse_args();run(a.run_id,a.mode,a.device,a.resume,a.stage)
