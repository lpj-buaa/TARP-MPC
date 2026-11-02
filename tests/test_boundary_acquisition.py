import numpy as np
import pytest
from types import SimpleNamespace


def test_recovery_gap_respects_power_limit_and_charge_efficiency():
    from tarp37.mpc import LP
    from tarp_boundary.domain import restore_gap
    lp=LP();p=lp.var(0.,1.,cost=1.)
    result=restore_gap(lp,[{p:.95}],[.2],[1.8],ceiling=2.)
    assert result['known'] and result['gap_mwh']==pytest.approx(.65)
    assert not result['recoverable']


def test_recovery_gap_is_zero_when_reserve_is_reachable_within_budget():
    from tarp37.mpc import LP
    from tarp_boundary.domain import restore_gap
    lp=LP();p=lp.var(0.,1.,cost=1.)
    assert restore_gap(lp,[{p:.95}],[1.],[1.8],2.)['recoverable']
    assert restore_gap(lp,[{p:.95}],[1.],[1.8],.1)['gap_mwh']==pytest.approx(.705)


def test_unverified_recovery_solver_does_not_create_a_boundary_label():
    from tarp37.mpc import LP
    from tarp_boundary.domain import restore_gap
    lp=LP();p=lp.var(0.,1.)
    result=restore_gap(lp,[{p:.95}],[1.],[1.8],2.,solve=lambda x:SimpleNamespace(success=False,status=4,message='numerical'))
    assert not result['known'] and result['gap_mwh'] is None


def test_adjacent_boundaries_require_known_both_sides_and_same_context():
    from tarp_boundary.metrics import adjacent_boundaries
    rows=[dict(id=str(i),key='a',family='release',dose=i/3,recoverable=r) for i,r in enumerate([[True]*3,[True]*3,[False]*3,None])]
    rows.append(dict(id='b',key='b',family='release',dose=.5,recoverable=[False]*3))
    edges=adjacent_boundaries(rows)
    assert edges==[('1','2',0),('1','2',1),('1','2',2)]


def test_boundary_reward_uses_queried_same_family_evidence_and_excludes_self():
    from tarp_boundary.learning import information_reward
    x=dict(id='x',key='a',family='release',dose=.6,descriptor=[1.],recoverable=[False]*3,turnover=[False]*4)
    y=dict(id='y',key='a',family='release',dose=.5,descriptor=[0.],recoverable=[True]*3,turnover=[True]*4)
    assert information_reward(x,[x])==0.
    assert information_reward(x,[x,y])>information_reward(x,[x,{**y,'key':'b'}])


def test_reference_target_never_uses_candidate_resources():
    from tarp_boundary.domain import reference_target
    ref={'view':{'ess_energy_mwh':[1.,1.2,1.5,1.8]}}
    np.testing.assert_array_equal(reference_target(ref),[1.,1.2,1.5,1.8])


def test_pair_refinement_uses_observed_bracket_and_public_doses_only():
    from tarp_boundary.learning import bracket_proposal
    known=[dict(id='left',key='a',family='reference',dose=0.,recoverable=[True]*3),
           dict(id='right',key='a',family='release',dose=1.,recoverable=[False]*3)]
    pool=[dict(id=str(d),key='a',family='release',dose=d) for d in [.1,.4,.8]]
    assert bracket_proposal(pool,known)['id']=='0.4'
    assert bracket_proposal(pool,[{**x,'key':'b'} for x in known]) is None


def test_all_acquisition_arms_charge_unique_answers_and_balance_contexts():
    from tarp_boundary.learning import acquire
    from tarp_info.cache import Oracle
    initial=[dict(id='ref'+k,key=k,family='reference',dose=0.,state=[0.],action=[0.]*51,
        descriptor=[0.],recoverable=[True]*3,turnover=[False]*4,error=0.) for k in ['a','b']]
    public=[dict(id=k+str(j),key=k,family='release',dose=j/5,state=[0.],action=[j/5]*51) for k in ['a','b'] for j in range(1,5)]
    answers={p['id']:dict(descriptor=[p['dose']],recoverable=[False]*3,turnover=[False]*4,error=1.) for p in public}
    for method in ['random','sobol','bracket_random']:
        oracle=Oracle(answers);r=acquire(public,initial,oracle,method,6,42)
        assert len(set(oracle.calls))==6 and [x['key'] for x in r['queries']]==['a','b']*3


def test_random_query_order_ignores_hidden_future_values():
    from tarp_boundary.learning import acquire
    from tarp_info.cache import Oracle
    public=[dict(id=str(j),key='a',family='release',dose=j/5,state=[0.],action=[j/5]*51) for j in range(1,5)]
    initial=[dict(id='ref',key='a',family='reference',dose=0.,state=[0.],action=[0.]*51,descriptor=[0.],recoverable=[True]*3,turnover=[False]*4,error=0.)]
    def run(sign):
        answers={p['id']:dict(descriptor=[sign*p['dose']],recoverable=[sign>0]*3,turnover=[False]*4,error=sign) for p in public}
        return acquire(public,initial,Oracle(answers),'random',3,42)['queries']
    assert [x['id'] for x in run(1)]==[x['id'] for x in run(-1)]


def test_boundary_localization_counts_two_revealed_adjacent_actions():
    from tarp_boundary.metrics import evaluate
    ref=dict(id='ref',key='a',family='reference',dose=0.,descriptor=[0.],recoverable=[True]*3,turnover=[False]*4,event=[0]*4,event_known=[True]*4)
    endpoint={**ref,'id':'end','family':'release','dose':1.,'descriptor':[1.],'recoverable':[False]*3}
    public=[dict(id='l',key='a',family='release',dose=1/3),dict(id='r',key='a',family='release',dose=2/3)]
    fields=['descriptor','recoverable','turnover','event','event_known']
    answers={'l':{**{k:ref[k] for k in fields},'descriptor':[.3]},'r':{**{k:endpoint[k] for k in fields},'descriptor':[.7]}}
    bundle=dict(initial=[ref,endpoint],public=public,answers=answers)
    result=evaluate(bundle,dict(queries=[{**public[0],**answers['l']},{**public[1],**answers['r']}],budget=2))
    assert result[0]['boundary_pairs_found']==0
    assert result[-1]['boundary_pairs_found']==1 and result[-1]['boundary_pairs_total']==1


def test_knowledge_summary_uses_only_revealed_same_context_and_excludes_self():
    from tarp_boundary.learning import knowledge_features
    x=dict(id='x',key='a',family='release',dose=.5,state=[0.],action=[.5]*51)
    ref=dict(id='ref',key='a',family='reference',dose=0.,state=[0.],action=[0.]*51,recoverable=[True]*3,turnover=[False]*4)
    right=dict(id='r',key='a',family='release',dose=1.,state=[0.],action=[1.]*51,recoverable=[False]*3,turnover=[True]*4)
    k=knowledge_features(x,[ref,right]);assert len(k)==18
    np.testing.assert_array_equal(k,knowledge_features(x,[ref,right,{**x,'recoverable':[False]*3,'turnover':[True]*4}]))
    np.testing.assert_array_equal(k,knowledge_features(x,[ref,right,{**right,'key':'b','id':'other'}]))


def test_reference_left_boundary_has_the_action_family_bracket_width():
    from tarp_boundary.metrics import evaluate
    ref=dict(id='ref',key='a',family='reference',dose=0.,descriptor=[0.],recoverable=[True]*3,turnover=[False]*4)
    endpoint={**ref,'id':'end','family':'release','dose':1.,'descriptor':[1.],'recoverable':[False]*3}
    public=[dict(id='x',key='a',family='release',dose=1/13)]
    answer=dict(descriptor=[.1],recoverable=[False]*3,turnover=[False]*4)
    result=evaluate(dict(initial=[ref,endpoint],public=public,answers={'x':answer}),dict(budget=1,queries=[dict(public[0],**answer)]))
    assert result[-1]['boundary_bracket_width']==pytest.approx(1/13)
