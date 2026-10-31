import numpy as np


def test_mpc_and_executor_share_generator_and_network_bounds():
    from tarp37.network import UnifiedNetwork
    from tarp37.mpc import BusinessMPC
    n=UnifiedNetwork(); m=BusinessMPC(n)
    v=dict(charge_cap_mw=np.r_[np.full(47,.05),np.ones(4)],discharge_cap_mw=np.r_[np.zeros(47),np.ones(4)],
           need_mwh=np.full(47,.02),due_hour=np.ones(47),ess_energy_mwh=np.ones(4),
           previous_power_mw=np.zeros(51),arrival_prior_mw=np.zeros((6,47)))
    forecast=dict(values=np.tile([16.,2.,4.,35.],(6,1)))
    plan=m.plan(v,forecast,np.full(4,.25),np.full(2,6.))
    assert plan['success']
    assert np.asarray(plan['planned_generation_mw']).shape==(6,10)
    assert np.max(plan['planned_generation_mw'])<=30+1e-6
    assert np.max(np.abs(plan['planned_q_mvar']))<=30+1e-6
    assert np.min(plan['planned_voltage_pu'])>=.93-1e-6
    assert np.max(plan['planned_voltage_pu'])<=1.07+1e-6
    assert plan['physics_version']==n.c.manifest()['version']
    r=n.dispatch(forecast['values'][0],plan['power_mw'],np.full(2,6.))
    assert r['physical_valid']
    assert 'risk' not in plan['objective_terms']


def test_dynamic_preferences_are_observation_based_and_normalized():
    from tarp37.mpc import update_preferences
    w=update_preferences([.25]*4,[.25]*4,price_pressure=.2,service_pressure=.9,resource_pressure=.5,change_pressure=.1)
    assert np.isclose(w.sum(),1) and (w>0).all()
    assert w[1]>w[0]


def test_balanced_policy_does_not_starve_existing_fast_charge_demand_on_healthy_grid():
    from tarp37.network import UnifiedNetwork
    from tarp37.mpc import BusinessMPC
    n=UnifiedNetwork(); m=BusinessMPC(n)
    view=dict(charge_cap_mw=np.r_[np.zeros(37),np.full(10,.2),np.ones(4)],
        discharge_cap_mw=np.r_[np.zeros(47),np.ones(4)],need_mwh=np.r_[np.zeros(37),np.full(10,.1)],
        due_hour=np.ones(47),ess_energy_mwh=np.ones(4),previous_power_mw=np.zeros(51),arrival_prior_mw=np.zeros((6,47)))
    plan=m.plan(view,dict(values=np.tile([16.,1.,0.,30.],(6,1))),np.full(4,.25),np.full(2,6.))
    assert sum(plan['power_mw'][37:47])>.8, 'native generation cost scale must not economically force universal service starvation'
