from copy import deepcopy
import numpy as np
import pytest


def test_action_bank_has_32_finite_distinct_actions():
    from tarp_sequence.corpus import action_bank
    base=np.ones(51)*.1;view=dict(charge_cap_mw=[1.]*51,discharge_cap_mw=[.3]*51,
                                 v2g_available_mwh=[.2]*47,need_mwh=[.4]*47,due_hour=list(range(47)))
    rows=action_bank(base,view,np.ones(47))
    assert len(rows)==32
    assert [r['family'] for r in rows[:3]]==['reference','throttle','release']
    assert all(np.asarray(r['power']).shape==(51,) and np.isfinite(r['power']).all() for r in rows)
    assert len({tuple(np.round(r['power'],8)) for r in rows})==32


def test_context_identity_separates_source_hour_and_preference():
    from tarp_sequence.corpus import context_key
    r=dict(source_id='s',pool_hour=32,profile_id=0)
    assert context_key(r)!=context_key(dict(r,pool_hour=40))
    assert context_key(r)!=context_key(dict(r,profile_id=1))


def test_mdp_budget_and_reward_telescope():
    from tarp_sequence.environment import AcquisitionEnv
    responses=np.arange(5*22,dtype=float).reshape(5,22)/100
    case=dict(descriptor=responses,valid=np.ones(5,bool),recoverable=np.array([[True]*3,[True]*3,[False]*3,[False]*3,[True]*3]),
              boundary_pairs=[(1,2)],initial=[0,4])
    env=AcquisitionEnv(case,2);env.reset();u0=env.utility();r1=env.step(1);r2=env.step(2)
    assert env.done and env.remaining_budget==0
    assert r1+r2==pytest.approx(env.utility()-u0)
    with pytest.raises(ValueError):env.step(2)


def test_gae_includes_future_reward_only_in_sequence_mode():
    from tarp_sequence.policy import gae
    rewards=[0.,0.,1.];values=[0.,0.,0.]
    a,_=gae(rewards,values,1.,1.);b,_=gae(rewards,values,0.,1.)
    np.testing.assert_allclose(a,[1,1,1]);np.testing.assert_allclose(b,[0,0,1])


def test_set_policy_is_invariant_to_observation_order():
    import torch
    from tarp_sequence.policy import SetPolicy,feature_tensors
    from tarp_sequence.environment import AcquisitionEnv
    case=dict(state=np.zeros(823),action=np.zeros((5,51)),doses=[0,.2,.4,.6,1],initial=[0,4],
              response=np.arange(300).reshape(5,60)/300,response_mask=np.ones((5,60),bool),
              descriptor=np.zeros((5,22)),valid=np.ones(5,bool),recoverable=np.ones((5,3),bool),boundary_pairs=[])
    env=AcquisitionEnv(case,2);pred=np.zeros((5,63));f=feature_tensors(case,env,pred)
    net=SetPolicy();a,v=net({k:torch.tensor(x)[None] for k,x in f.items()})
    f['observations']=f['observations'][::-1].copy();f['observation_mask']=f['observation_mask'][::-1].copy()
    b,w=net({k:torch.tensor(x)[None] for k,x in f.items()})
    torch.testing.assert_close(a,b);torch.testing.assert_close(v,w)


def test_policy_features_never_read_unqueried_responses():
    from tarp_sequence.policy import feature_tensors
    from tarp_sequence.environment import AcquisitionEnv
    c=dict(state=np.zeros(823),action=np.zeros((5,51)),doses=[0,.2,.4,.6,1],initial=[0,4],
           response=np.zeros((5,60)),response_mask=np.ones((5,60),bool),descriptor=np.zeros((5,22)),
           valid=np.ones(5,bool),recoverable=np.ones((5,3),bool),boundary_pairs=[])
    d=deepcopy(c);d['response'][1:4]=999;d['descriptor'][1:4]=99
    a=feature_tensors(c,AcquisitionEnv(c,2),np.zeros((5,63)))
    b=feature_tensors(c,AcquisitionEnv(d,2),np.zeros((5,63)))
    for k in a:np.testing.assert_array_equal(a[k],b[k])


def test_response_loss_ignores_unknown_targets():
    import torch
    from tarp_sequence.response import continuous_loss
    p=torch.ones((2,60));y=torch.zeros_like(p);mask=torch.ones_like(p,dtype=torch.bool);mask[:,36:39]=False
    a=continuous_loss(p,y,mask);y[:,36:39]=999
    assert continuous_loss(p,y,mask)==a


def test_unknown_query_cannot_create_recovery_boundary():
    from tarp_sequence.data import boundary_pairs
    items=[dict(family='reference',dose=0),dict(family='release',dose=.5),dict(family='release',dose=1)]
    assert boundary_pairs(items,[True,False,True],np.array([[True]*3,[False]*3,[False]*3]))==[]


def test_policy_observes_absolute_budget_and_verified_boundary_flags():
    from tarp_sequence.policy import feature_tensors
    from tarp_sequence.environment import AcquisitionEnv
    c=dict(state=np.zeros(823),action=np.zeros((5,51)),doses=[0,.2,.4,.6,1],initial=[0,4],
           response=np.zeros((5,60)),response_mask=np.ones((5,60),bool),descriptor=np.zeros((5,22)),
           valid=np.ones(5,bool),recoverable=np.ones((5,3),bool),recovery_mask=np.ones((5,3),bool),boundary_pairs=[])
    a=feature_tensors(c,AcquisitionEnv(c,1),np.zeros((5,63)))
    b=feature_tensors(c,AcquisitionEnv(c,2),np.zeros((5,63)))
    assert not np.array_equal(a['budget'],b['budget'])
    d=deepcopy(c);d['recoverable'][4,0]=False
    e=feature_tensors(c,AcquisitionEnv(d,1),np.zeros((5,63)))
    assert not np.array_equal(a['observations'],e['observations'])


def test_local_baseline_has_no_learned_prediction_fallback():
    from tarp_sequence.baselines import local_predictions
    from tarp_sequence.environment import AcquisitionEnv
    c=dict(key='s',ids=['a','b'],state=np.zeros(823),action=np.zeros((2,51)),families=['reference','release'],doses=[0,1],initial=[0],
           response=np.zeros((2,60)),response_mask=np.zeros((2,60),bool),descriptor=np.zeros((2,22)),valid=np.zeros(2,bool),
           recoverable=np.zeros((2,3),bool),recovery_mask=np.zeros((2,3),bool),boundary_pairs=[])
    a=local_predictions(c,AcquisitionEnv(c,1),np.zeros((2,63)))
    b=local_predictions(c,AcquisitionEnv(c,1),np.ones((2,63)))
    np.testing.assert_array_equal(a,b)


def test_local_baseline_preserves_resource_threshold_instead_of_linear_binary_labels():
    from tarp_sequence.baselines import local_predictions
    from tarp_sequence.environment import AcquisitionEnv
    response=np.zeros((3,60));response[1,24:36]=1.2;response[1,36:39]=.95*.2*4/6.4
    c=dict(key='s',ids=['ref','end','query'],state=np.zeros(823),action=np.array([[0.]*51,[1.]*51,[.7]*51]),
           families=['reference','release','release'],doses=[0,1,.7],initial=[0,1],response=response,
           response_mask=np.ones((3,60),bool),descriptor=np.zeros((3,22)),valid=np.ones(3,bool),
           recoverable=np.array([[True]*3,[False]*3,[True]*3]),recovery_mask=np.ones((3,3),bool),boundary_pairs=[])
    prediction=local_predictions(c,AcquisitionEnv(c,1),np.zeros((3,63)))
    np.testing.assert_allclose(prediction[2,60:63],0.)
