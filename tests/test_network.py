import numpy as np


def test_ac_dispatch_restores_all_native_generation_and_independent_balance():
    from tarp37.network import UnifiedNetwork
    net = UnifiedNetwork()
    exo = np.array([16., 1., 3., 35.])
    r = net.dispatch(exo, np.zeros(51), np.full(2,6.))
    assert r['physical_valid'], r.get('status')
    assert len(r['generators']) == 10
    assert r['local_generation_mw'] > 0
    assert r['max_bus_residual_mva'] < 1e-5
    actual_re=np.zeros(66)
    for gen in r['renewable_generators']: actual_re[net.ix[gen['bus']]]+=gen['used_mw']
    expected=net.load_share*exo[0]-actual_re-net.backup_map@r['backup_by_feeder_mw']
    assert np.allclose(r['net_p'],expected,atol=1e-10), 'published spatial injections must follow actual site dispatch'
    assert r['max_line_loading_ratio'] <= 1+1e-6
    assert min(r['voltage']) >= .93-1e-6 and max(r['voltage']) <= 1.07+1e-6
    assert all(g['p_mw'] <= 30+1e-6 and abs(g['q_mvar']) <= 30+1e-6 for g in r['generators'])


def test_signed_ev_and_solar_do_not_delete_reactive_load():
    from tarp37.network import UnifiedNetwork
    net = UnifiedNetwork()
    p = np.zeros(51); p[0] = -1.
    first = net.injections(np.array([12.,0.,0.,30.]),np.zeros(51))
    second = net.injections(np.array([12.,4.,10.,30.]),p)
    assert np.allclose(first[1],second[1])
    assert np.isclose(first[0].sum()-second[0].sum(),15.)


def test_spatial_curtailment_is_not_redistributed_when_publishing_bus_state():
    from tarp37.network import UnifiedNetwork
    net=UnifiedNetwork(); exo=np.array([16.,1.,30.,35.])
    r=net.dispatch(exo,np.zeros(51),np.full(2,6.))
    assert r['physical_valid']
    actual=np.zeros(66)
    for g in r['renewable_generators']: actual[net.ix[g['bus']]]+=g['used_mw']
    expected=net.load_share*exo[0]*r['base_fraction']-actual-net.backup_map@r['backup_by_feeder_mw']
    assert np.allclose(r['net_p'],expected,atol=1e-10)


def test_finite_protection_can_shed_real_load_without_fictitious_supply():
    from tarp37.network import UnifiedNetwork
    net = UnifiedNetwork()
    r = net.dispatch(np.array([400.,0.,0.,30.]), np.zeros(51), np.zeros(2))
    assert r['physical_valid'], r.get('status')
    assert r['non_ev_shed_mwh'] > 0 and r['backup_mwh'] == 0
    assert r['recovery_cost'] > 0
    assert max(g['p_mw'] for g in r['generators']) <= 30+1e-6
    assert r['max_bus_residual_mva'] < 1e-5
