import numpy as np
import pytest


def test_protection_is_chosen_before_delivery_and_traffic_advances_once(tmp_path,monkeypatch):
    from tarp37.environment import UnifiedEnvironment
    env=UnifiedEnvironment(tmp_path/'causal',date='2024-04-01')
    calls=[]; actual=env.traffic.advance_hour
    def counted(request): calls.append(np.array(request)); return actual(request)
    monkeypatch.setattr(env.traffic,'advance_hour',counted)
    try:
        plan=env.plan(); row=env.step(plan['power_mw'],plan)
        assert len(calls)==1, 'execution must not replay future arrivals to select past control'
        assert row['protection_information']=='visible_forecast_before_native_delivery'
        assert row['recourse_semantics']=='hourly_quasi_steady_fixed_EV_delivery'
    finally: env.close()


def test_failed_post_delivery_recourse_does_not_erase_delivery(tmp_path,monkeypatch):
    from tarp37.environment import UnifiedEnvironment,ExecutionUnknown
    env=UnifiedEnvironment(tmp_path/'unknown',date='2024-04-01')
    try:
        plan=env.plan()
        original=env.network.dispatch
        def dispatch(*args,**kwargs):
            if kwargs.get('allow_shedding',True): return dict(physical_valid=False,converged=False,status='forced_test_unknown')
            return original(*args,**kwargs)
        monkeypatch.setattr(env.network,'dispatch',dispatch)
        monkeypatch.setattr(env.regulator,'review',lambda *args:dict(known=False,physical_valid=False,point=None,status='forced_Q_unknown'))
        with pytest.raises(ExecutionUnknown) as error: env.step(plan['power_mw'],plan)
        assert env.traffic.inst.ctime==3600
        assert 'station_mean_mw' in error.value.row['traffic']
        assert error.value.row['service_known']
        assert not error.value.row['physical_valid']
    finally: env.close()
