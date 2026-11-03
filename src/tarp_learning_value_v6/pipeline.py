from pathlib import Path
import argparse,time
import numpy as np
from tarp_boundary.io import PROJECT,read,write,sha,bind
from tarp_acq_evidence.pipeline import seal,verify_seal
from tarp_sequence.data import load_cases
from .data import role,partition,select_cases,response_scale,selected_rows
from .training import fit_network
from .learner import encode,load_bank,reward_truth
from .policy import fit_policy
from .experiment import METHODS,selections,fit_students,hybrid_decision


def prepare(root,mode,device):
    root=Path(root);source=PROJECT/'artifacts/sequence_interim/sequence_interim_s42_v1'
    verify_seal(source/'dataset_complete.json')
    cases=select_cases(load_cases(source),mode)
    for c in cases:
        if not c['response_mask'].all() or not c['recovery_mask'].all() or not c['valid'].all():
            raise ValueError('This reuse protocol requires the verified complete bank')
    root.mkdir(parents=True,exist_ok=True)
    report=Path(__file__).parent/'report.py'
    code={str(p.resolve()):sha(p) for p in (PROJECT/'src').rglob('*.py') if p.resolve()!=report.resolve()}
    launcher=PROJECT/'scripts/run_learning_value_v6.ps1';code[str(launcher.resolve())]=sha(launcher)
    bind(root/'run_protocol.json',dict(mode=mode,device=device,seeds=[42] if mode=='smoke' else [42,43,44],
        source=str(source.resolve()),dataset_sha=sha(source/'dataset.json'),code=code,
        query_budgets=[8,16,24],network_epochs=2 if mode=='smoke' else 100,policy_batches=2 if mode=='smoke' else 120,
        episodes_per_batch=2 if mode=='smoke' else 8,network_parameters=136639,policy_parameters=121794,
        regularization=1.,function_draws=4,reference_probe_coordinates=16,initial_exposure_only=True,
        primary_test_original_role='cal',scope='Existing source-held-out reuse; no fresh formal confirm',
        new_simulations=0,old_rl_training=0,old_expert_training=0))
    exposure=dict(cases=[dict(key=c['key'],source_id=c['source_id'],original_role=c['role'],role=role(c),**partition(c)) for c in cases],
        roles={r:sorted({c['source_id'] for c in cases if role(c)==r}) for r in ['train','tune','test','development']},
        scale=response_scale(cases).tolist())
    bind(root/'exposure_protocol.json',exposure)
    return cases,read(root/'run_protocol.json'),np.array(exposure['scale'])


def verify(root):
    root=Path(root);p=read(root/'run_protocol.json')
    if sha(Path(p['source'])/'dataset.json')!=p['dataset_sha']:raise ValueError('Frozen dataset changed')
    for path,h in p['code'].items():
        if sha(path)!=h:raise ValueError('Frozen code changed: '+path)
    for path in [*root.glob('priors/*/complete.json'),*root.glob('policies/*/complete.json'),*root.glob('selections/*/complete.json'),
        *root.glob('students/*/complete.json'),root/'evaluation_complete.json']:
        if path.exists():verify_seal(path)
    if (root/'models_frozen.json').exists():
        frozen=read(root/'models_frozen.json')
        for path,h in frozen['models'].items():
            if sha(path)!=h:raise ValueError('Frozen model or exposure changed: '+path)


def run(run_id,mode='reuse',device='cuda',resume=False,stage='all'):
    if Path(run_id).name!=run_id:raise ValueError('RunId must be a basename')
    root=PROJECT/'artifacts/learning_value_v6'/run_id
    if root.exists() and not resume:raise ValueError('Run exists; use --resume or a new RunId')
    cases,p,scale=prepare(root,mode,device);verify(root)
    if stage in ['prepare','verify']:return dict(stage=stage,verified=True)
    executed=[];skipped=[];counts=dict(prior_fits=0,policy_fits=0,student_fits=0);start=time.perf_counter()
    # Only training and tuning rewards are materialized before model freeze.
    truths={r:reward_truth(cases,r) for r in ['train','tune']}
    mapping=[]
    for seed in p['seeds']:
        prior=root/f'priors/s{seed}';marker=prior/'complete.json'
        if marker.exists():verify_seal(marker);skipped.append(f'prior_{seed}')
        else:
            print(f'prior s{seed}: initial branches only',flush=True)
            fit_network(cases,selected_rows(cases),prior,scale,seed,device,epochs=p['network_epochs'])
            encode(cases,prior/'model.pt',scale,prior/'encoding.npz',device)
            seal(marker,[prior/'model.pt',prior/'fit.json',prior/'encoding.npz']);counts['prior_fits']+=1;executed.append(f'prior_{seed}')
        encoded,refs,noise=load_bank(cases,prior/'encoding.npz')
        policy=root/f'policies/s{seed}';marker=policy/'complete.json'
        if marker.exists():verify_seal(marker);skipped.append(f'policy_{seed}')
        else:
            fit_policy(cases,encoded,refs,truths,scale,noise,policy,seed,device,p['policy_batches'],p['episodes_per_batch'])
            seal(marker,[policy/'model.pt',policy/'fit.json']);counts['policy_fits']+=1;executed.append(f'policy_{seed}')
        selected=[]
        for method in METHODS:
            target=root/f'selections/{method}_s{seed}';marker=target/'complete.json'
            if marker.exists():verify_seal(marker);skipped.append(f'select_{method}_{seed}')
            else:
                print(f'collect cached labels: {method} s{seed}',flush=True)
                rows=selections(cases,encoded,refs,truths,scale,noise,policy/'model.pt',method,seed,device)
                write(target/'rows.json',dict(rows=rows));seal(marker,[target/'rows.json']);executed.append(f'select_{method}_{seed}')
            selected.extend(read(target/'rows.json')['rows'])
        items,n=fit_students(root,cases,selected,scale,seed,device,p['network_epochs']);mapping.extend(items);counts['student_fits']+=n
    decision=hybrid_decision(mapping);bind(root/'hybrid_decision.json',decision)
    if decision['triggered']:
        print('Tuning condition triggered the prespecified PPO + bracket hybrid',flush=True)
        for method in ['hybrid','greedy_bracket']:
            for seed in p['seeds']:
                prior=root/f'priors/s{seed}';encoded,refs,noise=load_bank(cases,prior/'encoding.npz')
                target=root/f'selections/{method}_s{seed}';marker=target/'complete.json'
                if marker.exists():verify_seal(marker);skipped.append(f'select_{method}_{seed}')
                else:
                    rows=selections(cases,encoded,refs,truths,scale,noise,root/f'policies/s{seed}/model.pt',method,seed,device)
                    write(target/'rows.json',dict(rows=rows));seal(marker,[target/'rows.json']);executed.append(f'select_{method}_{seed}')
                items,n=fit_students(root,cases,read(target/'rows.json')['rows'],scale,seed,device,p['network_epochs'])
                mapping.extend(items);counts['student_fits']+=n
    bind(root/'student_manifest.json',dict(models=mapping))
    frozen_files=[root/'student_manifest.json',root/'hybrid_decision.json',root/'exposure_protocol.json']
    frozen_files.extend(Path(i['folder'])/'model.pt' for i in mapping)
    frozen_files.extend(root/f'policies/s{s}/model.pt' for s in p['seeds'])
    bind(root/'models_frozen.json',dict(models={str(f.resolve()):sha(f) for f in frozen_files},
        independent_test_labels_used_for_choice=0,hybrid_decided_before_test=True))
    marker=root/'evaluation_complete.json'
    if marker.exists():verify_seal(marker);skipped.append('evaluate')
    else:
        from .analysis import evaluate
        print('All models frozen; evaluate held-out sources without test adaptation',flush=True)
        evaluate(root,cases,scale,device)
        from .diagnostics import run_diagnostics
        run_diagnostics(root,cases,scale,device)
        seal(marker,[root/'evaluation.json',root/'diagnostics.json']);executed.append('evaluate')
    verify(root)
    status=dict(run_id=run_id,executed=executed,skipped=skipped,**counts,
        new_simulations=0,old_rl_training=0,old_expert_training=0,seconds=time.perf_counter()-start)
    write(root/'invocation_status.json',status);print(status,flush=True);return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='lv_reuse_s42_v1');p.add_argument('--mode',choices=['smoke','reuse'],default='reuse')
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true');p.add_argument('--stage',choices=['all','prepare','verify'],default='all')
    a=p.parse_args();run(a.run_id,a.mode,a.device,a.resume,a.stage)
