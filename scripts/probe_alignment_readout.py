"""Post-fit diagnostic only: frozen policies and observed-only greedy readouts."""
from pathlib import Path
import argparse
import numpy as np
import torch
from tarp_boundary.io import read,write,sha
from tarp_sequence.data import load_cases
from tarp_sequence.environment import AcquisitionEnv
from tarp_sequence.baselines import marginal_scores
from tarp_sequence.policy import SetPolicy,feature_tensors,stack_features
from tarp_alignment_v5.core import correct_predictions,marginal_teacher
from tarp_alignment_v5.analysis import common_environment,summarize


def probe(root):
    root=Path(root);source=Path(read(root/'run_protocol.json')['source'])
    cases=[c for c in load_cases(root) if c['role']=='development_test']
    torch.set_num_threads(1);rows=[];sensitivity=[];baseline=read(source/'evaluation.json')['rows']
    for seed in [42,43,44]:
        with np.load(source/f'response_s{seed}/predictions.npz') as f:base={k:f[k] for k in f.files}
        with np.load(root/f'response_s{seed}/predictions.npz') as f:learned={k:f[k] for k in f.files}
        policies={}
        for name in ['ppo','rank','rank_correct']:
            path=source/f'policies/ppo_s{seed}/model.pt' if name=='ppo' else root/f'policies/{name}_s{seed}/model.pt'
            model=SetPolicy();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);model.eval();policies[name]=model
        for c in cases:
            for budget in [8,16,24]:
                for name in ['static','fixed','learned']:
                    prior=learned[c['key']] if name=='learned' else base[c['key']]
                    env=AcquisitionEnv(c,budget)
                    while not env.done:
                        p=prior if name=='static' else correct_predictions(c['action'],prior,env.observations())
                        score=marginal_scores(c,env,p);available=np.flatnonzero(env.available())
                        env.step(int(available[np.argmax(score[available])]),score=False)
                    if name=='static':
                        original=next(r for r in baseline if r['method']=='greedy' and r['seed']==seed and r['key']==c['key'] and r['budget']==budget)
                        if original['queries']!=env.queries:raise ValueError('Original static greedy was not reproduced')
                    rows.append(dict(method=name,seed=seed,source_id=c['source_id'],key=c['key'],role=c['role'],budget=budget,**env.metrics()))
            for extra in [0,4,8]:
                env=common_environment(c,extra);gain,_=marginal_teacher(env);available=np.flatnonzero(env.available())
                for name,model in policies.items():
                    prior=learned[c['key']] if name=='rank_correct' else base[c['key']]
                    normal=correct_predictions(c['action'],prior,env.observations()) if name=='rank_correct' else prior
                    variants={'normal':normal,'zero_candidate_consequence':normal.copy(),'oracle_candidate_consequence':normal.copy(),
                        'zero_candidate_all':normal.copy(),'oracle_candidate_all':normal.copy()}
                    variants['zero_candidate_consequence'][:,:60]=0
                    variants['oracle_candidate_consequence'][:,:60]=c['response']
                    variants['zero_candidate_all'][:,:63]=0
                    variants['oracle_candidate_all'][:,:60]=c['response']
                    variants['oracle_candidate_all'][:,60:63]=~c['recoverable'].astype(bool)
                    result={}
                    with torch.no_grad():
                        for variant,p in variants.items():
                            score,_=model(stack_features([feature_tensors(c,env,p)],'cpu'))
                            values=score.numpy()[0,available]
                            result[variant]=dict(choice=int(available[values.argmax()]),centered=values-values.mean())
                    ordinary=result['normal']
                    for variant in variants.keys()-{'normal'}:
                        altered=result[variant]
                        sensitivity.append(dict(source_id=c['source_id'],key=c['key'],seed=seed,method=name,history_extra=extra,variant=variant,
                            choice_changed=altered['choice']!=ordinary['choice'],
                            true_gain_difference=float(gain[altered['choice']]-gain[ordinary['choice']]),
                            centered_score_relative_change=float(np.linalg.norm(altered['centered']-ordinary['centered'])/max(np.linalg.norm(ordinary['centered']),1e-12)),
                            input_response_to_action_norm=float(np.linalg.norm(normal[available,:60]/np.sqrt(60),axis=1).mean()/np.linalg.norm(c['action'][available],axis=1).mean())))
    summary={method:{variant:dict(choice_changed=float(np.mean([r['choice_changed'] for r in sensitivity if r['method']==method and r['variant']==variant])),
        mean_relative_score_change=float(np.mean([r['centered_score_relative_change'] for r in sensitivity if r['method']==method and r['variant']==variant])),
        input_response_to_action_norm=float(np.mean([r['input_response_to_action_norm'] for r in sensitivity if r['method']==method and r['variant']==variant])))
        for variant in ['zero_candidate_consequence','oracle_candidate_consequence','zero_candidate_all','oracle_candidate_all']} for method in policies}
    write(root/'readout_probe.json',dict(rows=rows,summary=summarize(rows),sensitivity=sensitivity,sensitivity_summary=summary,
        scope='Post-fit exploratory mechanism probe; oracle inputs are privileged diagnostics, never an operational method',
        analysis_sha=sha(root/'analysis.json'),script_sha=sha(__file__),new_simulations=0,new_training=0))
    print(summarize(rows),flush=True);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);probe(p.parse_args().root)
