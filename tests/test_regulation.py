from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandapower as pp


def sample_row():
    root = Path(__file__).resolve().parents[2] / 'tarp_37_risk_audit/artifacts/runs/risk37_formal_v4_s42_20261101'
    rec = json.loads((root / 'sources/2024-03-04_s101/records.json').read_text())[0]
    return rec['rows'][0]


def test_q_regulation_restores_a_saved_violation_without_active_redispatch():
    from tarp37.regulation import RoutineQRegulator
    from tarp37.network import UnifiedNetwork
    from tarp37.data import ForecastTrace
    row = sample_row(); point = deepcopy(row['routine_operating_point'])
    n = UnifiedNetwork()
    result = RoutineQRegulator(n).review(ForecastTrace().actual(row['now']), row['actual_device_mw'], point)
    assert result['known'] and result['physical_valid']
    out = result['point']
    assert out['voltage_excess_pu'] <= n.c.voltage_tolerance_pu
    assert out['max_bus_residual_mva'] < 1e-5
    for j, g in enumerate(n.gen):
        if g['bus'] not in n.roots:
            assert out['generators'][j]['p_mw'] == point['generation_mw'][j]
    assert point == row['routine_operating_point'], 'Never mutate saved commands'


def test_failed_regulation_remains_unknown_instead_of_a_failure_event(monkeypatch):
    from tarp37.regulation import RoutineQRegulator
    from tarp37.network import UnifiedNetwork
    from tarp37.data import ForecastTrace
    row = sample_row(); regulator = RoutineQRegulator(UnifiedNetwork())
    def fail(*args):
        raise pp.OPFNotConverged('diagnostic numerical failure')
    monkeypatch.setattr(regulator, '_solve', fail)
    out = regulator.review(ForecastTrace().actual(row['now']), row['actual_device_mw'], row['routine_operating_point'])
    assert not out['known'] and out['point'] is None
    assert out['residual_voltage_event'] is None


def test_extra_cost_requires_a_valid_normal_baseline_and_is_signed():
    from tarp37.regulation import extra_recourse
    actual = dict(total_operating_cost=8., generation_cost=6., recovery_cost=2.,
                  generators=[dict(p_mw=1., q_mvar=.2)], wind_curtail_mwh=0., solar_curtail_mwh=0.,
                  backup_mwh=0., non_ev_shed_mwh=0., physical_valid=True)
    normal = dict(actual, total_operating_cost=10.)
    delta = extra_recourse(dict(known=True, physical_valid=True, point=normal), actual)
    assert delta['operating_cost_change'] == -2.
    missing = extra_recourse(dict(known=False, physical_valid=False, point=None), actual)
    assert missing['operating_cost_change'] is None
    assert missing['actual_operating_cost'] == 8.


def test_q_regulation_does_not_dispose_active_surplus_as_line_heat(monkeypatch):
    from types import SimpleNamespace
    from tarp37.regulation import RoutineQRegulator
    pre = dict(known=True, voltage_excess_pu=.01, pcc_export_permission_excess_mw=2.)
    network = SimpleNamespace(c=SimpleNamespace(voltage_tolerance_pu=1e-6, power_tolerance_mw=1e-6),
                              pressure=lambda *args:pre)
    regulator = RoutineQRegulator(network)
    monkeypatch.setattr(regulator, '_solve', lambda *args: (_ for _ in ()).throw(AssertionError('No Q heating workaround')))
    result = regulator.review([1.,0.,0.,30.], [0.]*51, {})
    assert not result['known']
    assert result['status'] == 'active_export_boundary_requires_separate_recours'
