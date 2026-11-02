from copy import deepcopy
import numpy as np
import pytest
from tarp_acq_evidence.features import spatial_raw, memory, state_input, PHYSICAL_DIM, MEMORY_DIM
from tarp_acq_evidence.data import stable_response


def row(ident='a',dose=.5,key='s'):
    return dict(id=ident,key=key,family='release',dose=dose,state=[0.]*49,spatial=[0.]*(PHYSICAL_DIM-49),
                action=[dose]*51,descriptor=[0.]*22,response=[0.]*60,recoverable=[True]*3,turnover=[False]*4)


def test_spatial_distinguishes_redistribution_hidden_by_aggregate():
    ctx=dict(bus_voltage_pu=[1.]*66,bus_net_p_mw=[0.]*66,bus_q_mvar=[0.]*66,
             view=dict(charge_cap_mw=[1.]*51,discharge_cap_mw=[0.]*51,need_mwh=[1.]*47,
                       due_hour=[2.]*47,v2g_available_mwh=[0.]*47,previous_power_mw=[0.]*51,arrival_prior_mw=[[0.]*47]*6))
    other=deepcopy(ctx);other['view']['need_mwh'][0]+=1;other['view']['need_mwh'][1]-=1
    assert np.sum(ctx['view']['need_mwh'])==np.sum(other['view']['need_mwh'])
    assert not np.array_equal(spatial_raw(ctx),spatial_raw(other))


def test_quantitative_memory_senses_values_without_binary_change():
    x=row('q',.4);a=row('seen',0);b=deepcopy(a);b['response'][0]=.25
    assert np.array_equal(memory(x,[a],False),memory(x,[b],False))
    assert not np.array_equal(memory(x,[a],True),memory(x,[b],True))


def test_memory_excludes_self_and_other_context():
    x=row(); seen=row('b',0); poison=deepcopy(x);poison['response']=[99.]*60
    foreign=row('c',.2,'other');foreign['response']=[99.]*60
    np.testing.assert_array_equal(memory(x,[seen],True),memory(x,[seen,poison,foreign],True))
    assert len(memory(x,[seen],True))==MEMORY_DIM
    assert len(state_input(x,True))==PHYSICAL_DIM


def test_model_parameter_match_and_response_memory_independence():
    import torch
    from tarp_acq_evidence.model import Agent
    a=Agent(42,False);b=Agent(42,True)
    assert sum(p.numel() for p in a.net.parameters())==sum(p.numel() for p in b.net.parameters())
    s=torch.zeros(2,PHYSICAL_DIM);act=torch.zeros(2,51);m=torch.randn(2,MEMORY_DIM)
    _,r1=a.net(s,act,m);_,r2=a.net(s,act,m+10)
    assert torch.equal(r1,r2)


def test_local_interpolation_uses_only_known_rows():
    from tarp_acq_evidence.experiment import interpolate
    x=row('q',.5);a=row('left',0);b=row('right',1)
    a['response'][24:36]=[0.]*12;b['response'][24:36]=[2.]*12
    z=interpolate(x,[a,b]);np.testing.assert_allclose(z['response'][24:36],1.)
    assert z['id']=='q' and z['recoverable']==[True]*3


def test_matched_history_is_identical_and_query_budget_unique():
    from tarp_acq_evidence.experiment import experiment
    from tarp_info.cache import Oracle
    initial=[row('ref',0)];initial[0]['family']='reference'
    allrows=[row(str(i),i/5) for i in range(1,5)]
    private={r['id']:{k:r[k] for k in ['response','descriptor','recoverable','turnover']} for r in allrows}
    public=[{k:v for k,v in r.items() if k not in ['response','descriptor','recoverable','turnover']} for r in allrows]
    fixed=['3','1','4']
    results=[experiment(public,initial,Oracle(private),v,3,42,fixed, warm=1,updates=1,checkpoints=[2]) for v in ['old_summary','quantitative','stable','combined']]
    for r in results:
        assert [x['id'] for x in r['queries']]==fixed
        assert r['rankings'][0]['candidate_ids']==['2','4']
        assert r['oracle_calls']==fixed


def test_output_seal_detects_corruption(tmp_path):
    from tarp_acq_evidence.pipeline import seal,verify_seal
    p=tmp_path/'result.json';p.write_text('{}')
    seal(tmp_path/'complete.json',[p]);verify_seal(tmp_path/'complete.json')
    p.write_text('{"changed":1}')
    with pytest.raises(ValueError,match='hash'):verify_seal(tmp_path/'complete.json')


def test_cached_continuous_targets_reconstruct_frozen_descriptor(tmp_path):
    from pathlib import Path
    from tarp_acq_evidence.data import prepare
    from tarp_acq_evidence.experiment import from_response
    source=Path('artifacts/boundary_v2/boundary_formal_s42_v2')
    if not source.exists():pytest.skip('local frozen cache not installed')
    bundle=prepare(source,tmp_path/'bundle.json')
    rows=bundle['initial']+[dict(p,**bundle['answers'][p['id']]) for p in bundle['public']]
    for r in rows:
        derived=from_response(r,r['response'])
        np.testing.assert_allclose(derived['descriptor'],r['descriptor'],atol=1e-10,rtol=0)
        assert len(r['response'])==60
    assert all('response' not in p and 'descriptor' not in p for p in bundle['public'])


def test_constant_score_has_zero_discrimination_not_dropped_source():
    from tarp_acq_evidence.experiment import ranking_metrics
    assert ranking_metrics(['a','b','c'],[1,1,1],[0,.1,.2],['s']*3)[0]['spearman']==0


def test_source_bootstrap_averages_seeds_before_resampling():
    from tarp_acq_evidence.analysis import paired
    records=[]
    for source,delta in [('a',1.),('b',3.)]:
        for seed in [42,43,44]:
            records.extend([dict(method='a',seed=seed,source_id=source,value=0.),
                            dict(method='b',seed=seed,source_id=source,value=delta)])
    result=paired(records,'b','a')
    assert result['n_sources']==2 and result['difference']==2


def test_first_active_decision_independent_of_unqueried_answers():
    from tarp_acq_evidence.experiment import experiment
    from tarp_info.cache import Oracle
    initial=[row('ref',0)];initial[0]['family']='reference'
    allrows=[row(str(i),i/5) for i in range(1,5)]
    private={r['id']:{k:r[k] for k in ['response','descriptor','recoverable','turnover']} for r in allrows}
    changed=deepcopy(private)
    for r in changed.values():r['response']=[100.]*60;r['descriptor']=[100.]*22
    public=[{k:v for k,v in r.items() if k not in ['response','descriptor','recoverable','turnover']} for r in allrows]
    a=experiment(public,initial,Oracle(private),'combined',1,42,warm=1,updates=1,checkpoints=[])
    b=experiment(public,initial,Oracle(changed),'combined',1,42,warm=1,updates=1,checkpoints=[])
    assert a['queries'][0]['id']==b['queries'][0]['id']
