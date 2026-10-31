import numpy as np


def test_native_power_limit_and_snapshot_reproduce_actual_delivery(tmp_path):
    from tarp37.traffic import NativeTraffic
    traffic=NativeTraffic(tmp_path/'native',participation=.5)
    try:
        # 1 hour is a mechanical interface test; research pools use 24h warmup.
        traffic.advance_hour(np.full(47,.03))
        snapshot=traffic.snapshot(tmp_path/'snapshot')
        limited=traffic.advance_hour(np.full(47,.01))
        traffic.restore(snapshot)
        repeated=traffic.advance_hour(np.full(47,.01))
        assert np.allclose(limited['station_mean_mw'],repeated['station_mean_mw'],atol=1e-10)
        assert max(limited['station_mean_mw'])<=.01+1e-9
        assert abs(limited['battery_balance_error_kwh'])<1e-5
        assert len(limited['quarters'])==4
        assert traffic.inst.ctime==7200 and traffic.inst.core.W.world.TIME==7200
    finally: traffic.close()


def test_zero_power_wait_is_finite_and_does_not_schedule_in_the_past():
    from tarp37.traffic import safe_retry_time
    assert safe_retry_time(3600,0,10,0,10)==3610
    assert safe_retry_time(3600,0,10,1/3600,10)>3600


def test_departure_retry_keeps_peak_gap_and_separate_trip_identity():
    from tarp37.traffic import aggregate_departure_gaps
    events=[dict(kind='departure_delay',vehicle='ev1',trip_id=2,gap_kwh=5.),
            dict(kind='departure_delay',vehicle='ev1',trip_id=2,gap_kwh=.0001),
            dict(kind='departure_delay',vehicle='ev1',trip_id=3,gap_kwh=2.)]
    assert aggregate_departure_gaps(events)==7.


def test_signed_native_v2g_is_grid_power_with_conversion_loss(tmp_path):
    from tarp37.traffic import NativeTraffic
    t=NativeTraffic(tmp_path/'signed',participation=1.)
    try:
        t.advance_hour(np.zeros(47))
        station=max(list(t.inst.scs),key=lambda s:len(s._chi))
        assert len(station._chi)>0
        station_index=t.c.station_names.index(station.name)
        for v in station._chi: v._energy=.9*v._cap
        snapshot=t.snapshot(tmp_path/'signed_snapshot')
        request=np.zeros(47); request[station_index]=-.01
        first=t.advance_hour(request)
        assert abs(first['station_mean_mw'][station_index]+.01)<1e-8
        assert abs(first['discharging_battery_kwh']-10/.85)<1e-6
        assert abs(first['battery_balance_error_kwh'])<1e-5
        t.restore(snapshot); repeated=t.advance_hour(request)
        assert np.allclose(first['station_mean_mw'],repeated['station_mean_mw'],atol=1e-10)
    finally: t.close()
