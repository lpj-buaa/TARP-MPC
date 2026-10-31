import numpy as np


def executed_point(network, result):
    source = {s: np.zeros(66) for s in ('wind', 'solar')}
    for g in result['renewable_generators']:
        source[g['source']][network.ix[g['bus']]] += g['used_mw']
    return dict(generation_mw=[g['p_mw'] for g in result['generators']],
        q_mvar=[g['q_mvar'] for g in result['generators']],
        renewable_used_by_source_bus_mw={s: x.tolist() for s, x in source.items()},
        backup_mw=result['backup_by_feeder_mw'], base_fraction=result['base_fraction'])


def test_pre_correction_ac_reconstructs_complete_executed_point():
    from tarp37.network import UnifiedNetwork
    n = UnifiedNetwork(); exo = np.array([16., 1., 30., 35.]); power = np.zeros(51)
    post = n.dispatch(exo, power, np.full(2, 6.), allow_shedding=False)
    assert post['physical_valid']
    pre = n.pressure(exo, power, executed_point(n, post))
    assert pre['converged'] and pre['physical_valid']
    assert np.allclose(pre['voltage'], post['voltage'], atol=2e-7)
    assert np.allclose(pre['loading_ratio'], post['loading_ratio'], atol=2e-7)
    assert np.isclose(pre['solar_used_mw'], post['solar_used_mw'], atol=1e-8)
    assert np.isclose(pre['wind_used_mw'], post['wind_used_mw'], atol=1e-8)
    assert pre['voltage_excess_pu'] <= 1e-6
    assert pre['max_bus_residual_mva'] < 1e-5


def test_complete_plan_honors_fixed_q_backup_shedding_and_available_renewables():
    from tarp37.network import UnifiedNetwork
    n = UnifiedNetwork(); exo = np.array([16., 1., 3., 35.]); power = np.zeros(51)
    post = n.dispatch(exo, power, np.full(2, 6.))
    point = executed_point(n, post)
    point['q_mvar'][1] += .1
    point['base_fraction'] = .9
    point['backup_mw'] = [.2, .3]
    point['renewable_used_by_source_bus_mw']['solar'] = (n.solar_share * 10.).tolist()
    pre = n.pressure(exo, power, point)
    assert pre['converged']
    for j, g in enumerate(n.gen):
        if g['bus'] not in n.roots:
            assert np.isclose(pre['generators'][j]['q_mvar'], point['q_mvar'][j], atol=1e-9)
    assert np.allclose(pre['backup_by_feeder_mw'], [.2, .3])
    assert np.isclose(pre['solar_used_mw'], 3.)
    assert np.allclose(pre['net_p'], n.load_share * 16. * .9 -
        np.array(pre['renewable_used_by_bus_mw']) - n.backup_map @ [.2, .3])


def test_capacity_separates_reverse_permission_from_thermal_overload():
    from tarp37.network import UnifiedNetwork
    n = UnifiedNetwork(); exo = np.array([16., 0., 0., 35.])
    point = dict(generation_mw=[0.] * 10, q_mvar=[0.] * 10,
        renewable_used_by_source_bus_mw=dict(wind=[0.] * 66, solar=[0.] * 66),
        backup_mw=[0., 0.], base_fraction=1.)
    for j, g in enumerate(n.gen):
        if g['bus'] not in n.roots: point['generation_mw'][j] = 3.
    pre = n.pressure(exo, np.zeros(51), point)
    assert pre['converged'] and pre['pcc_export_permission_excess_mw'] > 0
    assert pre['capacity_ratio_peak'] < 1., 'negative root P is not line/import overload'
    assert pre['pcc_import_excess_mw'] == 0.


def test_mpc_publishes_all_routine_controls_for_ac_verification():
    from tarp37.network import UnifiedNetwork
    from tarp37.mpc import BusinessMPC, routine_point
    n = UnifiedNetwork()
    view = dict(charge_cap_mw=np.r_[np.full(47, .05), np.ones(4)],
        discharge_cap_mw=np.r_[np.zeros(47), np.ones(4)], need_mwh=np.zeros(47),
        due_hour=np.ones(47), ess_energy_mwh=np.ones(4), previous_power_mw=np.zeros(51),
        arrival_prior_mw=np.zeros((6, 47)))
    forecast = dict(values=np.tile([16., 1., 30., 35.], (6, 1)))
    plan = BusinessMPC(n).plan(view, forecast, np.full(4, .25), np.full(2, 6.))
    point = routine_point(plan)
    assert np.asarray(plan['planned_solar_used_by_bus_mw']).shape == (6, 66)
    assert np.asarray(plan['planned_wind_used_by_bus_mw']).shape == (6, 66)
    for h in range(6):
        used = np.array(plan['planned_solar_used_by_bus_mw'][h]) + plan['planned_wind_used_by_bus_mw'][h]
        assert np.allclose(used + plan['planned_curtailment_mw'][h], n.solar_share * 30. + n.wind_share)
    assert 0. <= point['base_fraction'] <= 1.
    assert n.pressure(forecast['values'][0], plan['power_mw'], point)['converged']


def test_mpc_breaks_unpriced_reactive_circulation_ties_without_changing_primary_objective():
    from tarp37.network import UnifiedNetwork
    from tarp37.mpc import BusinessMPC
    n=UnifiedNetwork()
    view=dict(charge_cap_mw=np.r_[np.full(47,.05),np.ones(4)],
        discharge_cap_mw=np.r_[np.zeros(47),np.ones(4)],need_mwh=np.zeros(47),
        due_hour=np.ones(47),ess_energy_mwh=np.ones(4),previous_power_mw=np.zeros(51),
        arrival_prior_mw=np.zeros((6,47)))
    plan=BusinessMPC(n).plan(view,dict(values=np.tile([16.,1.,30.,35.],(6,1))),np.full(4,.25),np.full(2,6.))
    load_q=float(np.sum(n.load_share*16.*n.q_ratio))
    assert np.max(np.abs(plan['planned_q_mvar']).sum(axis=1)) < 2*load_q
    assert plan['reactive_tiebreak']['primary_objective_increase'] <= 1e-6
