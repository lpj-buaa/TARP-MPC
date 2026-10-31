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
