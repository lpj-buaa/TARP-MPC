import numpy as np


def test_hourly_closed_loop_never_changes_delivered_energy_after_ac(tmp_path):
    from tarp37.environment import UnifiedEnvironment
    env=UnifiedEnvironment(tmp_path/'loop',date='2024-04-01')
    try:
        view=env.view(); forecast=env.forecast()
        plan=env.controller.plan(view,forecast,env.weights,env.backup)
        row=env.step(plan['power_mw'],plan)
        assert row['physical_valid']
        assert np.allclose(row['actual_device_mw'][:47],row['traffic']['station_mean_mw'])
        assert row['battery_balance_error_kwh']<1e-5
        assert row['ess_balance_error_mwh']<1e-8
        assert np.allclose(env.backup,np.full(2,6)-row['backup_by_feeder_mw'])
        assert env.hour==1 and env.traffic.inst.ctime==3600
        from tarp37.contract import RISK_VERSION
        assert row['risk_contract']==RISK_VERSION
        assert row['pre_known']
    finally: env.close()
