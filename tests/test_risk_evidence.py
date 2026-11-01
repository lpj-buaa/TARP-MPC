import numpy as np


def normal_row():
    return dict(pre_known=True,pre_voltage_excess_pu=0.,pre_capacity_ratio_peak=.8,
        regulation_known=True,regulated_voltage_excess_pu=0.,
        pre_voltage=[.94,1.06],pre_pcc_export_permission_excess_mw=2.,
        service_known=True,departure_energy_deficit_kwh=0.,fcs_wait_max_minutes=0.,
        wind_available_mwh=1.,solar_available_mwh=1.,wind_curtail_mwh=0.,solar_curtail_mwh=0.)


def test_quality_deviation_and_reverse_direction_do_not_create_voltage_or_thermal_events():
    from tarp37.risks import window_labels
    rows=[normal_row() for _ in range(6)]
    event,known=window_labels(rows)
    assert known.all() and not event.any()


def test_shared_contract_controls_service_and_sourcewise_curtailment_readout():
    from tarp37.contract import PhysicalContract
    from tarp37.risks import window_labels,intensities
    c=PhysicalContract.load(); c.waiting_sla_minutes=45.; c.renewable_utilization_target=.8
    rows=[normal_row() for _ in range(6)]
    for r in rows:
        r.update(fcs_wait_max_minutes=40.,wind_available_mwh=.1,wind_curtail_mwh=.05,
            solar_available_mwh=10.,solar_curtail_mwh=0.)
    event,known=window_labels(rows,contract=c)
    assert event.tolist()==[0.,0.,0.,1.] and known.all()
    assert np.isclose(intensities(rows[0],contract=c)[3],.5), 'same sourcewise ratio as event definition'


def test_corrections_are_measured_relative_to_complete_routine_plan():
    from tarp37.evidence import correction_evidence
    pre=dict(known=True,converged=True,voltage_excess_pu=.01,capacity_ratio_peak=.8,
        pcc_export_permission_excess_mw=0.,generator_bound_excess=0.,
        generators=[dict(p_mw=1.,q_mvar=.2)],wind_used_mw=1.,solar_used_mw=2.,
        backup_by_feeder_mw=[.1,.2],base_fraction=.9)
    post=dict(physical_valid=True,generators=[dict(p_mw=1.2,q_mvar=.4)],
        wind_used_mw=.8,solar_used_mw=2.,backup_by_feeder_mw=[.2,.2],base_fraction=.8,
        protection_attempts=[dict(status='numerical_unknown'),dict(status='valid')])
    e=correction_evidence(pre,post,[10.,1.,2.,30.])
    assert e['triggered_constraints']==['voltage']
    assert np.isclose(e['additional_non_ev_shed_mwh'],1.)
    assert np.isclose(e['additional_backup_mwh'],.1)
    assert np.isclose(e['wind_used_change_mwh'],-.2)
    assert e['numerical_fallback_used'] and not e['shedding_necessity_proven']


def test_unknown_pre_flow_does_not_claim_a_physical_trigger():
    from tarp37.evidence import correction_evidence
    e=correction_evidence(dict(known=False,converged=False),
        dict(physical_valid=False,protection_attempts=[]),[10.,1.,2.,30.])
    assert e['triggered_constraints']==[] and not e['correction_comparison_known']


def test_environment_keeps_routine_diagnostics_and_post_delivery_corrections_aligned(tmp_path,monkeypatch):
    from tarp37.environment import UnifiedEnvironment
    env=UnifiedEnvironment(tmp_path/'alignment',date='2024-04-01')
    captured=[]; original=env.network.pressure
    def capture(exo,power,routine):
        captured.append(routine)
        return original(exo,power,routine)
    monkeypatch.setattr(env.network,'pressure',capture)
    try:
        plan=env.plan(); row=env.step(plan['power_mw'],plan)
        assert all('q_mvar' in r and 'base_fraction' in r for r in captured)
        assert row['routine_operating_point']==captured[-1]
        assert row['pre_scope']=='complete_routine_plan_at_realized_delivery_before_extra_AC_recourse'
        assert row['correction_evidence']['actual_AC_valid']
        assert np.allclose(row['actual_device_mw'][:47],row['traffic']['station_mean_mw'])
        assert row['forecast_allocation_guard']['available_before_delivery']
        assert 'applied_diagnostic' in row['forecast_allocation_guard']
        assert row['forecast_allocation_guard']['forecast_values']==env.trace.visible(row['now'])['values'][0].tolist()
        assert 'voltage_deviation' not in row['risk_events_hour']
    finally: env.close()
