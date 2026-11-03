from pathlib import Path
import hashlib
import numpy as np
import torch
from tarp_boundary.io import read,write
from tarp_sequence.data import load_cases,GROUPS,TRAIN_ROLES,TUNE_ROLES
from tarp_sequence.environment import AcquisitionEnv
from tarp_sequence.baselines import marginal_scores,local_predictions
from tarp_sequence.response import predicted_descriptor
from tarp_sequence.policy import SetPolicy,feature_tensors,stack_features
from tarp_interim.analysis import source_macro,rank_summary
from .core import correct_predictions,marginal_teacher
from .policy import ARMS


def prediction_metrics(case,env,prediction):
    mask=np.array(case['response_mask'],bool)&env.available()[:,None]
    error=np.asarray(prediction)[:,:60]-case['response'];ref=case['families'].index('reference')
    delta_error=(prediction[:,:60]-prediction[ref,:60])-(case['response']-case['response'][ref])
    delta_mask=mask&case['response_mask'][ref]
    significant=delta_mask&(np.abs(case['response']-case['response'][ref])>.01)
    def average(x,m):return float(np.mean(x[m])) if m.any() else None
    desc=predicted_descriptor(prediction)
    valid=env.available()&case['valid']
    turn_match=(desc[:,18:]>0)==(case['descriptor'][:,18:]>0)
    return dict(mae=average(np.abs(error),mask),delta_mae=average(np.abs(delta_error),delta_mask),
        direction_accuracy=average((np.sign(prediction[:,:60]-prediction[ref,:60])==np.sign(case['response']-case['response'][ref])).astype(float),significant),
        turn_accuracy=float(turn_match[valid].mean()) if valid.any() else None,
        unknown_actions=int(env.available().sum()),known_target_components=int(mask.sum()),
        groups={g:average(np.abs(error[:,lo:hi]),mask[:,lo:hi]) for g,(lo,hi) in GROUPS.items()})


def summarize(rows):
    return {method:{str(b):{metric:source_macro([r for r in rows if r['method']==method and r['budget']==b],metric)
        for metric in ['utility','distance','boundary_recall']} for b in sorted({r['budget'] for r in rows})}
        for method in sorted({r['method'] for r in rows})}


def common_environment(case,extra):
    env=AcquisitionEnv(case,16)
    seed=int(hashlib.sha256(case['key'].encode()).hexdigest()[:8],16)
    history=np.random.default_rng(seed).permutation(np.where(env.available())[0])[:extra]
    for i in history:env.step(int(i),score=False)
    return env


def analyze(root,device='cpu'):
    root=Path(root);protocol=read(root/'run_protocol.json');source=Path(protocol['source'])
    cases=load_cases(root);rows=read(root/'evaluation.json')['rows'];regression=[];ranking=[]
    evaluable=[c for c in cases if c['role'] not in TRAIN_ROLES|TUNE_ROLES]
    for seed in protocol['seeds']:
        with np.load(source/f'response_s{seed}/predictions.npz') as f:base={k:f[k] for k in f.files}
        with np.load(root/f'response_s{seed}/predictions.npz') as f:conditional={k:f[k] for k in f.files}
        policies={}
        for arm in ['ppo',*ARMS]:
            path=source/f'policies/ppo_s{seed}/model.pt' if arm=='ppo' else root/f'policies/{arm}_s{seed}/model.pt'
            model=SetPolicy().to(device);model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model']);model.eval();policies[arm]=model
        for c in evaluable:
            for extra in [0,4,8]:
                e=common_environment(c,extra);actual,eligible=marginal_teacher(e);ix=np.flatnonzero(eligible)
                alternatives={'base':base[c['key']],'correct_fixed':correct_predictions(c['action'],base[c['key']],e.observations()),
                    'conditional_static':conditional[c['key']],
                    'correct':correct_predictions(c['action'],conditional[c['key']],e.observations()),
                    'local':local_predictions(c,e,base[c['key']])}
                identity=dict(seed=seed,key=c['key'],source_id=c['source_id'],role=c['role'],history_extra=extra)
                for method,p in alternatives.items():
                    regression.append(dict(identity,method=method,**prediction_metrics(c,e,p)))
                    values=marginal_scores(c,e,p)
                    ranking.append(dict(identity,method=method,kind='response_readout',**rank_summary(actual[ix],values[ix])))
                for arm,model in policies.items():
                    p=alternatives['base'] if arm in ['ppo','rank'] else alternatives['correct'] if ARMS[arm][2] else alternatives['correct_fixed']
                    with torch.no_grad():logits,_=model(stack_features([feature_tensors(c,e,p)],device))
                    ranking.append(dict(identity,method=arm,kind='policy',**rank_summary(actual[ix],logits.cpu().numpy()[0,ix])))
    summaries={role:summarize([r for r in rows if r['role']==role]) for role in sorted({r['role'] for r in rows})}
    contrasts={}
    pairs={'R_minus_PPO':('rank','ppo'),'C_minus_PPO':('correct','ppo'),
        'RC_minus_PPO':('rank_correct','ppo'),'RC_minus_R':('rank_correct','rank'),
        'RC_minus_C':('rank_correct','correct'),'C_learning_minus_fixed':('correct','correct_fixed'),
        'RC_learning_minus_fixed':('rank_correct','rank_correct_fixed')}
    development=summaries['development_test']
    for name,(a,b) in pairs.items():
        contrasts[name]={budget:dict(mean=development[a][budget]['utility']['mean']-development[b][budget]['utility']['mean'],
            sources={s:development[a][budget]['utility']['sources'][s]-v for s,v in development[b][budget]['utility']['sources'].items()})
            for budget in ['8','16','24']}
    interaction={b:development['rank_correct'][b]['utility']['mean']-development['rank'][b]['utility']['mean']-
        development['correct'][b]['utility']['mean']+development['ppo'][b]['utility']['mean'] for b in ['8','16','24']}
    fits=[read(p) for p in root.glob('response_s*/fit.json')];policy_fits=[read(p) for p in root.glob('policies/*/fit.json')]
    matched_training={}
    for seed in protocol['seeds']:
        own=[f for f in policy_fits if f['seed']==seed]
        signature={(f['episodes'],f['training_cache_queries'],f['optimizer_steps']) for f in own}
        if len(own)!=len(ARMS) or len(signature)!=1:raise ValueError('Ablation training budgets differ')
        matched_training[str(seed)]=list(next(iter(signature)))
        if protocol['mode']!='smoke':
            baseline=read(source/f'policies/ppo_s{seed}/fit.json')
            if baseline['episodes']!=own[0]['episodes'] or baseline['training_cache_queries']!=own[0]['training_cache_queries']:
                raise ValueError('Training query schedule differs from original PPO')
    analysis=dict(summary=summaries,contrasts=contrasts,interaction=interaction,
        diagnostic_regression={role:{method:{str(h):{m:source_macro([r for r in regression if r['role']==role and r['method']==method and r['history_extra']==h],m)
            for m in ['mae','delta_mae','direction_accuracy','turn_accuracy']} for h in [0,4,8]}
            for method in sorted({r['method'] for r in regression})} for role in sorted({r['role'] for r in regression})},
        diagnostic_ranking={role:{kind:{method:source_macro([r for r in ranking if r['role']==role and r['kind']==kind and r['method']==method],'spearman')
            for method in sorted({r['method'] for r in ranking if r['kind']==kind})} for kind in ['response_readout','policy']}
            for role in sorted({r['role'] for r in ranking})},
        training=dict(response_fits=len(fits),policy_fits=len(policy_fits),
            response_seconds=sum(f['elapsed_seconds'] for f in fits),policy_seconds=sum(f['elapsed_seconds'] for f in policy_fits),
            response_optimizer_steps=sum(f['optimizer_steps'] for f in fits),policy_optimizer_steps=sum(f['optimizer_steps'] for f in policy_fits),
            cache_query_replays=sum(f['training_cache_queries'] for f in policy_fits),
            response_parameter_counts=sorted({f['parameters'] for f in fits}),policy_parameter_counts=sorted({f['parameters'] for f in policy_fits}),
            matched_episode_query_step_counts=matched_training,
            new_simulations=0,old_rl_training=0,warning_training=0),
        dataset=read(root/'dataset.json')['counts'],original_confirm_used=0,
        scope='Development reuse; source macro, two held-out development sources; no model selected using these results')
    write(root/'diagnostics.json',dict(regression_rows=regression,ranking_rows=ranking))
    write(root/'analysis.json',analysis)
    print('analysis complete: '+str(analysis['training']),flush=True)
    return analysis
