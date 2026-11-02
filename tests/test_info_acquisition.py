from copy import deepcopy
import numpy as np
import pytest


def record(energy=1., complete=True):
    return dict(rows=[dict(physical_valid=True, ess_energy_mwh=[energy]*4,
        backup_energy_remaining_mwh=[6.,6.], candidate_request_mw=[0.]*51,
        weights=[.25]*4) for _ in range(6 if complete else 2)])


def test_directional_capability_changes_have_opposite_signs():
    from tarp_info.profiles import describe, GROUPS
    full=describe(record(1.5), np.ones(47)); low=describe(record(.5), np.ones(47))
    assert len(full)==sum(GROUPS)
    # First hour: four discharge-energy, four absorption-energy projections.
    assert np.all(full[:4]>low[:4])
    assert np.all(full[4:8]<low[4:8])


def test_unknown_suffix_and_unverified_physics_are_not_zero_profiles():
    from tarp_info.profiles import describe
    assert describe(record(complete=False), np.ones(47)) is None
    r=record(); r['rows'][2]['physical_valid']=False
    assert describe(r,np.ones(47)) is None


def test_profile_excludes_risk_labels_and_outcome_severity():
    from tarp_info.profiles import describe
    r=record(); changed=deepcopy(r); changed['event']=[1,1,1,1]
    for row in changed['rows']:row.update(fcs_wait_max_minutes=999.,wind_curtail_mwh=999.,loss_mwh=999.)
    np.testing.assert_array_equal(describe(r,np.ones(47)),describe(changed,np.ones(47)))


def test_oracle_hides_future_until_charged_and_rejects_duplicate_calls():
    from tarp_info.cache import Oracle
    o=Oracle({'a':{'profile':[1.],'event':[1,0,0,0]}})
    assert o.calls==[]
    assert o.query('a')['profile']==[1.]
    assert o.calls==['a']
    with pytest.raises(ValueError,match='already'):o.query('a')


def test_novelty_excludes_self_and_cannot_cross_policy_contexts():
    from tarp_info.acquire import novelty
    known={'ctx':{'a':np.array([0.]),'b':np.array([1.])}, 'other':{'x':np.array([.1])}}
    assert novelty(np.array([0.]),'ctx',known,exclude='a')==1.
    known['ctx']['c']=np.array([.2])
    assert novelty(np.array([0.]),'ctx',known,exclude='a')==pytest.approx(.2)


def test_round_robin_budget_and_same_candidate_pool():
    from tarp_info.acquire import run_arm
    from tarp_info.cache import Oracle
    public=[dict(id=f'{c}{i}',key=c,action=[i/10.]*51,state=[0.,1.]) for c in ['x','y'] for i in range(4)]
    answers={p['id']:dict(profile=[float(p['id'][1])],event=[0,0,1,1],known=[True]*4,error=1.,late_service=False) for p in public}
    initial=[dict(id='ref'+c,key=c,action=[0.]*51,state=[0.,1.],profile=[0.],error=0.) for c in ['x','y']]
    o=Oracle(answers); result=run_arm(public,initial,o,'random',6,42)
    assert len(o.calls)==6 and len(set(o.calls))==6
    assert [x['key'] for x in result['queries']]==['x','y']*3


def test_random_selection_does_not_depend_on_hidden_future():
    from tarp_info.acquire import run_arm
    from tarp_info.cache import Oracle
    public=[dict(id=str(i),key='x',action=[i/10.]*51,state=[1.]) for i in range(5)]
    initial=[dict(id='ref',key='x',action=[0.]*51,state=[1.],profile=[0.],error=0.)]
    def answers(sign):return {p['id']:dict(profile=[sign*int(p['id'])],event=[0]*4,known=[True]*4,error=sign,late_service=False) for p in public}
    a=run_arm(public,initial,Oracle(answers(1)),'random',3,42)
    b=run_arm(public,initial,Oracle(answers(-1)),'random',3,42)
    assert [q['id'] for q in a['queries']]==[q['id'] for q in b['queries']]


def test_q_retry_uses_finite_initializations_and_keeps_failure_unknown():
    from tarp_info.native import retry_q
    calls=[]
    def solve(init):
        calls.append(init)
        if init=='flat':raise RuntimeError('numerical')
        return dict(known=True,physical_valid=True)
    assert retry_q(solve)['normal_q_initialization']=='pf'
    assert calls==['flat','pf']
    def failed(init):raise RuntimeError('still numerical')
    with pytest.raises(RuntimeError,match='still numerical'):retry_q(failed)


def test_sobol_is_per_context_and_within_public_action_domain():
    from tarp_info.acquire import sobol_targets
    public=[dict(key=k,action=([v]*37+[v]*10+[v]*4)) for k in ['x','y'] for v in [0.,1.]]
    targets=sobol_targets(public,8,42)
    assert targets['x'].shape==(4,51)
    assert targets['y'].shape==(4,51)
    for points in targets.values():
        assert np.all(points>=0) and np.all(points<=1)
        assert points[:,0].min()<.5 and points[:,0].max()>.5
    assert not np.array_equal(targets['x'],targets['y'])


def test_protocol_hashes_real_legacy_learner_dependency():
    from tarp_info.cli import code_dependencies
    assert any(p.endswith('tarp37/acquisition.py') for p in code_dependencies())


def test_mpc_projection_keeps_original_lp_class_and_checks_layout():
    from pathlib import Path
    import json
    import tarp37.mpc as original
    from tarp37.network import UnifiedNetwork
    import importlib.util
    spec=importlib.util.spec_from_file_location('opportunity_probe',Path(__file__).resolve().parents[1]/'scripts/opportunity_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    project_options=module.project_options
    files=list((Path(__file__).resolve().parents[1]/'artifacts/native_information/v5_information_sanity_v1').glob('*/record.json'))
    if not files:pytest.skip('Native sanity context required for physical projection test')
    context=json.loads(files[0].read_text(encoding='utf-8'))['visible_views'][1]
    before=original.LP
    result=project_options(UnifiedNetwork(),context,.05)
    assert result['known'] and result['layout_verified']
    assert original.LP is before
    for bounds in result['bounds_mw'].values():assert bounds[1]>=bounds[0]-1e-6


def test_mpc_projection_never_reports_a_numerically_missing_endpoint_as_known():
    from pathlib import Path
    import json,importlib.util
    from types import SimpleNamespace
    from tarp37.network import UnifiedNetwork
    spec=importlib.util.spec_from_file_location('opportunity_probe',Path(__file__).resolve().parents[1]/'scripts/opportunity_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files=list((Path(__file__).resolve().parents[1]/'artifacts/native_information/v5_information_sanity_v1').glob('*/record.json'))
    if not files:pytest.skip('Native sanity context required')
    context=json.loads(files[0].read_text(encoding='utf-8'))['visible_views'][1]
    failed=lambda lp:SimpleNamespace(success=False,status=4,message='injected numerical failure')
    result=module.project_options(UnifiedNetwork(),context,.05,projection_solver=failed)
    assert not result['known'] and 'endpoint' in result['reason']
    assert result['endpoint_attempts'][0]['status']==4
