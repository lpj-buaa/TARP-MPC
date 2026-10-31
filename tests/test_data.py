import json
from pathlib import Path
import numpy as np


def test_export_reference_is_finite_and_uses_common_observed_hours():
    reference = json.loads((Path(__file__).resolve().parents[1]/'artifacts/data/reference.json').read_text(encoding='utf-8'))
    assert np.isfinite(list(reference.values())).all(), 'missing native actuals must not poison capacity sizing'
    assert reference['reference_common_hours'] > 5000


def test_visible_forecast_cannot_use_a_future_issue():
    from tarp37.data import ForecastTrace
    trace = ForecastTrace()
    now = trace.time_for_date('2024-04-01', 8)
    forecast = trace.visible(now)
    assert np.isfinite(forecast['values']).all()
    assert np.max(forecast['issued_at']) <= now
    trace.issues[trace.position(now),0,0] = now+1
    import pytest
    with pytest.raises(ValueError, match='publication'):
        trace.visible(now)


def test_actual_eia_renewables_use_hour_end_as_interval_end_not_start():
    from tarp37.data import ForecastTrace
    from tarp37.assets import PROJECT
    trace=ForecastTrace(); now=trace.time_for_date('2024-04-02',7)
    with np.load(PROJECT/'artifacts/data/hourly.npz') as cached:
        pos=int(np.searchsorted(cached['time_s'],now))
        end=int(np.searchsorted(cached['time_s'],now+3600))
        want=cached['actual'][pos].copy()
        want[1:3]=cached['actual'][end,1:3]
    want*=trace.scale
    want[1]=np.clip(want[1],0.,trace.wind_nameplate)
    want[2]=np.clip(want[2],0.,trace.solar_nameplate)
    assert np.allclose(trace.actual(now)[:3],want[:3]), 'EIA period T+1 describes interval [T,T+1), unlike CAISO interval-start'
