import copy
from pathlib import Path
import numpy as np
import pytest


def case_fixture():
    families=['reference','throttle','release']+['throttle']*12+['release']*12+['charging_only','storage_only','spatial_shift','fcs_priority','asymmetric_joint']
    doses=[0.,1.,1.]+list(np.arange(1,13)/13)*2+[1.]*5
    return dict(key='source_h32_p0',source_id='source',role='train',initial=[0,1,2],families=families,doses=doses,
        ids=[str(i) for i in range(32)],state=np.zeros(823),action=np.repeat(np.array(doses)[:,None],51,1),
        response=np.zeros((32,60)),response_mask=np.ones((32,60),bool),recoverable=np.ones((32,3),bool),
        recovery_mask=np.ones((32,3),bool),gaps=np.zeros((32,3)),valid=np.ones(32,bool),descriptor=np.zeros((32,22)),boundary_pairs=[])


def test_partition_is_public_disjoint_and_preserves_24_query_budget():
    from tarp_learning_value_v6.data import partition
    c=case_fixture();p=partition(c)
    assert len(p['initial'])==3 and len(p['audit'])==5 and len(p['query'])==24
    assert sorted(p['initial']+p['audit']+p['query'])==list(range(32))
    c['response'][:]=999;c['recoverable'][:]=False
    assert partition(c)==p


def test_task_error_keeps_the_worst_component_and_classification_error():
    from tarp_learning_value_v6.core import task_errors
    true=np.zeros((2,60));prediction=np.zeros((2,63));prediction[0,7]=3
    failure=np.zeros((2,3));prediction[1,60:]=20
    e=task_errors(prediction,true,failure,np.ones(60))
    assert e[0]==3 and e[1]>.999


def test_readout_update_matches_regularized_least_squares_and_is_set_invariant():
    from tarp_learning_value_v6.core import Readout
    rng=np.random.default_rng(4);h=rng.normal(size=(8,4));base=rng.normal(size=(8,63));targets=rng.normal(size=(8,63))
    state=Readout(h,base,1.)
    for i in [0,2,4]:state.observe(i,targets[i])
    w=np.linalg.solve(np.eye(4)+h[[0,2,4]].T@h[[0,2,4]],h[[0,2,4]].T@(targets[[0,2,4]]-base[[0,2,4]]))
    assert np.allclose(state.mean,base+h@w,atol=1e-10)
    other=Readout(h,base,1.)
    for i in [4,2,0]:other.observe(i,targets[i])
    assert np.allclose(state.mean,other.mean,atol=1e-10)


def test_learning_reward_can_be_negative_and_telescopes():
    from tarp_learning_value_v6.core import learning_reward
    losses=[2.,1.,1.5,.4]
    rewards=[learning_reward(a,b,losses[0]) for a,b in zip(losses,losses[1:])]
    assert rewards[1]<0 and sum(rewards)==pytest.approx(.8)


def test_student_rows_never_include_audit_or_nontraining_sources():
    from tarp_learning_value_v6.data import selected_rows,partition,fingerprint
    c=case_fixture();test=copy.deepcopy(c);test['role']='cal';test['key']='test_h32_p0'
    p=partition(c);selection={c['key']:p['query']}
    rows=selected_rows([c,test],selection)
    assert len(rows)==27 and all(x['role']=='train' for x,_ in rows)
    assert not any(i in p['audit'] for _,i in rows)
    assert fingerprint(rows)==fingerprint(list(reversed(rows)))
    with pytest.raises(ValueError,match='Audit'):
        selected_rows([c],{c['key']:p['audit']})


def test_episode_selector_never_reads_unqueried_or_audit_truth():
    from tarp_learning_value_v6.acquisition import Episode
    from tarp_learning_value_v6.data import partition
    c=case_fixture();rng=np.random.default_rng(3)
    encoded=dict(features=rng.normal(size=(32,5)),base=rng.normal(size=(32,63)))
    ref=dict(features=rng.normal(size=(6,5)),base=rng.normal(size=(6,63)),probe=list(range(6)))
    e=Episode(c,encoded,ref,np.ones(60),np.ones(63)*.1,8,42)
    score=e.values()[0];c2=copy.deepcopy(c)
    unseen=partition(c)['query']+partition(c)['audit'];c2['response'][unseen]=999;c2['recoverable'][unseen]=False
    other=Episode(c2,encoded,ref,np.ones(60),np.ones(63)*.1,8,42)
    assert np.allclose(score,other.values()[0])
    assert e.available().sum()==24
    for i in partition(c)['query'][:8]:e.step(i)
    assert e.done and len(e.queries)==8
    assert not set(e.queries)&set(partition(c)['audit'])


def test_zero_residual_equals_value_greedy_and_residual_bound_holds():
    import torch
    from tarp_learning_value_v6.policy import anchored_logits
    v=torch.tensor([[.1,.3,-.2]]);raw=torch.zeros_like(v);available=torch.ones_like(v,dtype=torch.bool)
    score=anchored_logits(raw,v,available)
    assert score.argmax(-1).item()==v.argmax(-1).item()
    raw=torch.tensor([[100.,-100.,100.]])
    score=anchored_logits(raw,v,available)
    chosen=score.argmax(-1).item()
    assert v.max()-v[0,chosen] <= 2*.25*v.abs().max()+1e-6


def test_hybrid_is_decided_on_tuning_without_test_labels(tmp_path):
    from tarp_learning_value_v6.experiment import hybrid_decision
    from tarp_boundary.io import write
    mapping=[]
    for method,value in [('value_greedy',1.),('value_ppo',1.01)]:
        folder=tmp_path/method;write(folder/'fit.json',dict(best_validation=value))
        mapping.extend(dict(method=method,budget=b,folder=str(folder)) for b in [8,16])
    result=hybrid_decision(mapping)
    assert result['triggered'] and not result['test_labels_used']


def test_independent_evaluation_refuses_unfrozen_models(tmp_path):
    from tarp_learning_value_v6.analysis import evaluate
    with pytest.raises(ValueError,match='freeze'):
        evaluate(tmp_path,[],np.ones(60),'cpu')


def test_recorded_and_update_distributions_are_identical():
    import torch
    from tarp_learning_value_v6.policy import policy_logits
    from tarp_sequence.policy import SetPolicy,stack_features
    from tarp_learning_value_v6.acquisition import Episode
    c=case_fixture();rng=np.random.default_rng(3)
    bank=dict(features=rng.normal(size=(32,4)),base=rng.normal(size=(32,63)))
    ref=dict(features=rng.normal(size=(5,4)),base=rng.normal(size=(5,63)),probe=list(range(5)))
    env=Episode(c,bank,ref,np.ones(60),np.ones(63)*.1,8,42)
    model=SetPolicy();features=stack_features([env.features()],'cpu')
    a,_=policy_logits(model,features);dist=torch.distributions.Categorical(logits=a);choice=dist.sample();logp=dist.log_prob(choice)
    b,_=policy_logits(model,features)
    assert torch.equal(logp,torch.distributions.Categorical(logits=b).log_prob(choice))
    assert sum(p.numel() for p in model.parameters())==121794


def test_checkpoint_selection_uses_absolute_error_not_easy_episode_ratios():
    from tarp_learning_value_v6.policy import validation_score
    a=[dict(source_id='a',final_loss=8.,relative_error=.8),dict(source_id='b',final_loss=1.,relative_error=1.)]
    b=[dict(source_id='a',final_loss=9.,relative_error=.9),dict(source_id='b',final_loss=.8,relative_error=.8)]
    assert validation_score(a)<validation_score(b)


def test_zero_ppo_hybrid_equals_greedy_with_identical_rule(tmp_path):
    import torch
    from tarp_sequence.policy import SetPolicy
    from tarp_learning_value_v6.experiment import selections
    c=case_fixture();rng=np.random.default_rng(4)
    encoded={c['key']:dict(features=rng.normal(size=(32,4)),base=rng.normal(size=(32,63)))}
    reference=dict(features=rng.normal(size=(5,4)),base=rng.normal(size=(5,63)),probe=list(range(5)))
    truth=dict(response=np.zeros((5,60)),failure=np.zeros((5,3)),sources=['source']*5)
    model=SetPolicy();torch.nn.init.zeros_(model.actor[-1].weight);torch.nn.init.zeros_(model.actor[-1].bias)
    path=tmp_path/'model.pt';torch.save(dict(model=model.state_dict()),path)
    first=selections([c],encoded,{'train':reference},{'train':truth},np.ones(60),np.ones(63)*.1,path,'hybrid',42,'cpu')
    second=selections([c],encoded,{'train':reference},{'train':truth},np.ones(60),np.ones(63)*.1,path,'greedy_bracket',42,'cpu')
    assert [r['queries'] for r in first]==[r['queries'] for r in second]


@pytest.mark.skipif(not Path('artifacts/sequence_interim/sequence_interim_s42_v1/dataset_complete.json').is_file(),reason='Frozen bank integration')
def test_protocol_binds_io_and_seal_runtime_dependencies(tmp_path):
    from tarp_learning_value_v6.pipeline import prepare
    _,p,_=prepare(tmp_path,'smoke','cpu')
    names=[Path(k).as_posix() for k in p['code']]
    assert any(k.endswith('/tarp_boundary/io.py') for k in names)
    assert any(k.endswith('/tarp_acq_evidence/pipeline.py') for k in names)


def test_exact_one_percent_improvement_does_not_trigger_hybrid(tmp_path):
    from tarp_learning_value_v6.experiment import hybrid_decision
    from tarp_boundary.io import write
    rows=[]
    for name,value in [('value_greedy',1.),('value_ppo',.99)]:
        p=tmp_path/name;write(p/'fit.json',dict(best_validation=value))
        rows.extend(dict(method=name,budget=b,folder=str(p)) for b in [8,16])
    assert not hybrid_decision(rows)['triggered']


def test_sensitivity_keeps_value_and_consequence_channels_separate():
    from tarp_learning_value_v6.diagnostics import feature_variants
    f=dict(candidates=np.ones((32,116),np.float32),observations=np.ones((32,177),np.float32),information=np.ones(32,np.float32))
    variants=feature_variants(f,np.arange(32))
    assert np.all(variants['without_candidate_outcomes']['candidates'][:,51:114]==0)
    assert np.all(variants['without_candidate_outcomes']['information']==1)
    assert np.all(variants['without_information']['information']==0)
    assert np.all(variants['without_information']['candidates'][:,51:114]==1)
    assert np.all(f['candidates']==1)
