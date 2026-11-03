import numpy as np
import pytest
import torch
from pathlib import Path


def small_case():
    return dict(key='toy',source_id='toy',role='train',state=np.zeros(823),
        action=np.repeat(np.linspace(0,1,32)[:,None],51,axis=1),
        response=np.zeros((32,60)),response_mask=np.ones((32,60),bool),
        gaps=np.zeros((32,3)),recoverable=np.ones((32,3),bool),recovery_mask=np.ones((32,3),bool),
        descriptor=np.zeros((32,22)),valid=np.ones(32,bool),initial=[0,16,31],boundary_pairs=[],
        families=['reference']+['throttle']*16+['release']*15,doses=list(np.linspace(0,1,32)))


def test_teacher_matches_actual_query_reward_including_two_endpoint_boundary():
    from tarp_alignment_v5.core import marginal_teacher
    from tarp_sequence.environment import AcquisitionEnv
    c = dict(descriptor=np.array([[0., 0.], [1., 0.], [0., 2.], [2., 1.]]),
             valid=np.ones(4, bool), initial=[0], boundary_pairs=[(1, 2), (0, 3)])
    for seen in [[0], [0, 1], [0, 1, 2]]:
        e = AcquisitionEnv(c, 3); e.seen = seen.copy()
        values, mask = marginal_teacher(e)
        before = e.utility()
        for i in np.flatnonzero(e.available()):
            e.seen.append(int(i)); expected = e.utility() - before; e.seen.pop()
            assert values[i] == pytest.approx(expected, abs=1e-12)
        assert np.array_equal(mask, e.available())


def test_observed_residual_correction_only_uses_queried_truth_and_masks():
    from tarp_alignment_v5.core import correct_predictions
    from tarp_sequence.environment import AcquisitionEnv
    action = np.array([[0.], [.5], [1.]])
    c = dict(action=action, response=np.ones((3, 60))*2, response_mask=np.ones((3,60), bool),
             recoverable=np.ones((3,3),bool), descriptor=np.zeros((3,22)), valid=np.ones(3,bool),
             initial=[0], boundary_pairs=[])
    c['response_mask'][0, 1] = False
    p = np.zeros((3,63)); p[:, 60:] = .37
    e = AcquisitionEnv(c, 2)
    out = correct_predictions(action, p, e.observations())
    assert np.allclose(out[:,0],2)
    assert np.allclose(out[:,1],0)
    assert np.array_equal(out[:,60:],p[:,60:])
    c['response'][1:] = -900
    assert np.array_equal(correct_predictions(action,p,e.observations()),out)
    assert np.array_equal(correct_predictions(action,p,[]),p)


def test_ranking_loss_trains_order_ignores_ties_and_invalid_candidates():
    from tarp_alignment_v5.core import ranking_loss
    score = torch.tensor([[0.,0.,500.]],requires_grad=True)
    truth = torch.tensor([[.4,.1,.9]])
    mask = torch.tensor([[True,True,False]])
    loss = ranking_loss(score,truth,mask); loss.backward()
    assert score.grad[0,0] < 0 < score.grad[0,1]
    assert score.grad[0,2] == 0
    assert ranking_loss(score.detach(),torch.ones_like(truth),mask).item() == 0


def test_residual_update_is_differentiable_with_no_added_parameters():
    from tarp_alignment_v5.core import residual_tensor
    from tarp_sequence.response import ResponseModel
    from tarp_sequence.policy import SetPolicy
    base = torch.zeros((1,2),requires_grad=True)
    prior = torch.tensor([[[1.,7.]]],requires_grad=True)
    true = torch.tensor([[[3.,9.]]])
    result = residual_tensor(base,prior,true,torch.tensor([[[True,False]]]),torch.ones((1,1)))
    assert torch.allclose(result,torch.tensor([[2.,0.]]))
    result.sum().backward()
    assert prior.grad[0,0,0] == -1 and prior.grad[0,0,1] == 0
    assert sum(p.numel() for p in ResponseModel().parameters()) == 136639
    assert sum(p.numel() for p in SetPolicy().parameters()) == 121794


def test_supervised_histories_exclude_target_and_only_use_same_training_context():
    from tarp_alignment_v5.training import training_neighbors
    train = [small_case(),small_case(),small_case()]
    offsets = np.cumsum([0]+[len(c['action']) for c in train])
    ids = np.array([0, 15, 31, 32, 80])
    neighbors, weights = training_neighbors(train, offsets, ids, np.random.default_rng(7))
    for i,row in zip(ids,neighbors):
        assert i not in row
        assert (np.searchsorted(offsets,row,side='right') == np.searchsorted(offsets,i,side='right')).all()
    assert np.allclose(weights.sum(1),1)
    with pytest.raises(ValueError,match='training'):
        training_neighbors([dict(train[0],role='development_test')],np.array([0,32]),[1],np.random.default_rng(7))


def test_zero_rank_coefficient_is_exact_original_ppo_update():
    from tarp_alignment_v5.policy import update_policy
    from tarp_sequence.policy import SetPolicy,update
    torch.manual_seed(4)
    a=SetPolicy(); b=SetPolicy(); b.load_state_dict(a.state_dict())
    features=dict(state=np.zeros(823,np.float32),candidates=np.zeros((32,116),np.float32),
        observations=np.zeros((32,177),np.float32),observation_mask=np.arange(32)<3,
        available=np.arange(32)>=3,budget=np.array([8/24,8/24],np.float32))
    histories=[[dict(features=features,action=3,logp=-3.3,value=.1,reward=.2)]]
    oa=torch.optim.Adam(a.parameters(),lr=3e-4);ob=torch.optim.Adam(b.parameters(),lr=3e-4)
    torch.manual_seed(8);update(a,oa,histories,'ppo','cpu')
    torch.manual_seed(8);update_policy(b,ob,histories,'cpu',rank_weight=0)
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p,q)


def test_teacher_is_loss_only_and_rejected_outside_training():
    from tarp_alignment_v5.policy import collect
    from tarp_sequence.policy import SetPolicy
    c=small_case();p=np.zeros((32,63))
    histories,_=collect(SetPolicy(),[c],[p],[1],'cpu',teacher=True)
    row=histories[0][0]
    assert 'gains' in row and 'gains' not in row['features']
    assert set(row['features']) == {'state','candidates','observations','observation_mask','available','budget'}
    with pytest.raises(ValueError,match='training'):
        collect(SetPolicy(),[dict(c,role='cal')],[p],[1],'cpu',teacher=True)


@pytest.mark.skipif(not Path('artifacts/sequence_interim/sequence_interim_s42_v1/dataset_complete.json').is_file(),
    reason='Sealed local dataset integration check; core unit tests use synthetic fixtures')
def test_prepare_reuses_exact_dataset_roles_and_rejects_modified_source(tmp_path):
    from tarp_alignment_v5.pipeline import prepare
    from tarp_boundary.io import read,sha
    source=Path('artifacts/sequence_interim/sequence_interim_s42_v1').resolve()
    prepare(source,tmp_path,'smoke','cpu')
    assert sha(source/'dataset.json')==sha(tmp_path/'dataset.json')
    protocol=read(tmp_path/'run_protocol.json')
    assert protocol['original_confirm_used']==0 and protocol['new_simulations']==0
    assert protocol['seeds']==[42]
    (tmp_path/'dataset.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='dataset'):
        prepare(source,tmp_path,'smoke','cpu')


def test_prediction_diagnostics_exclude_queried_targets():
    from tarp_alignment_v5.analysis import prediction_metrics
    from tarp_sequence.environment import AcquisitionEnv
    c=dict(action=np.zeros((3,51)),response=np.zeros((3,60)),response_mask=np.ones((3,60),bool),
        descriptor=np.zeros((3,22)),valid=np.ones(3,bool),initial=[0],boundary_pairs=[],families=['reference','x','x'])
    p=np.zeros((3,63));p[0,:60]=100;p[1:,:60]=2
    result=prediction_metrics(c,AcquisitionEnv(c,2),p)
    assert result['mae']==pytest.approx(2)
    assert result['unknown_actions']==2 and result['known_target_components']==120


def test_source_macro_does_not_count_repeated_actions_as_independent_sources():
    from tarp_alignment_v5.analysis import summarize
    rows=[dict(method='ppo',budget=8,source_id='a',utility=0.,distance=.4,boundary_recall=None)]*5
    rows+=[dict(method='ppo',budget=8,source_id='b',utility=1.,distance=.1,boundary_recall=.5)]
    result=summarize(rows)['ppo']['8']
    assert result['utility']['mean']==.5
    assert result['utility']['n_sources']==2


def test_verification_rejects_changed_analysis_code_and_evaluation(tmp_path):
    from tarp_alignment_v5.pipeline import verify
    from tarp_boundary.io import write,sha
    source=tmp_path/'source';source.mkdir();(source/'dataset.json').write_text('{}',encoding='utf-8')
    (tmp_path/'dataset.json').write_text('{}',encoding='utf-8')
    analysis=tmp_path/'analysis_source.py';analysis.write_text('version=1',encoding='utf-8')
    evaluation=tmp_path/'evaluation.json';evaluation.write_text('{}',encoding='utf-8')
    write(tmp_path/'run_protocol.json',dict(source=str(source),dependencies={},code={}))
    write(tmp_path/'analysis_protocol.json',dict(code={str(analysis):sha(analysis)},evaluation_sha=sha(evaluation)))
    verify(tmp_path)
    analysis.write_text('version=2',encoding='utf-8')
    with pytest.raises(ValueError,match='analysis'):verify(tmp_path)
    analysis.write_text('version=1',encoding='utf-8');evaluation.write_text('{"changed":true}',encoding='utf-8')
    with pytest.raises(ValueError,match='evaluation'):verify(tmp_path)
