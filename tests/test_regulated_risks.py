from copy import deepcopy
import numpy as np


def rows():
    return [dict(pre_known=True, pre_voltage_excess_pu=.03, pre_capacity_ratio_peak=.8,
                 regulation_known=True, regulated_voltage_excess_pu=0.,
                 service_known=True, departure_energy_deficit_kwh=0., fcs_wait_max_minutes=0.,
                 wind_available_mwh=1., solar_available_mwh=1., wind_curtail_mwh=0., solar_curtail_mwh=0.) for _ in range(6)]


def test_voltage_uses_normal_regulation_instead_of_raw_plan_pressure():
    from tarp37.risks import window_labels, intensities
    data = rows(); event, known = window_labels(data)
    assert known.all() and not event.any()
    assert intensities(data[0])[0] == 0.


def test_unknown_q_is_not_a_negative_but_an_observed_residual_is_positive():
    from tarp37.risks import window_labels
    data = rows(); data[0].update(regulation_known=False, regulated_voltage_excess_pu=None)
    event, known = window_labels(data)
    assert not event[0] and not known[0]
    data[1]['regulated_voltage_excess_pu'] = .001
    event, known = window_labels(data)
    assert event[0] and known[0]


def test_other_three_physical_heads_are_unchanged():
    from tarp37.risks import window_labels
    data = rows()
    for x in data:
        x.update(pre_capacity_ratio_peak=1.01, fcs_wait_max_minutes=31., wind_curtail_mwh=.2)
    event, known = window_labels(data)
    assert event.tolist() == [0., 1., 1., 1.] and known.all()


def test_hardware_matches_frozen_v4_manifest():
    from pathlib import Path
    import json
    from tarp37.contract import PhysicalContract
    previous = Path(__file__).resolve().parents[2] / 'tarp_37_risk_audit/artifacts/runs/risk37_formal_v4_s42_20261101/audit/contract.json'
    old = json.loads(previous.read_text()); new = PhysicalContract.load().manifest()
    for key in ['line_limit_ka', 'voltage_limits_pu', 'impedance_scale', 'root_capacity_mw',
                'generators', 'station_slots', 'station_power_envelope_mw', 'added_ess', 'emergency', 'numerical_tolerances']:
        assert new[key] == old[key], key


def test_online_feature_entry_rejects_v4_contract():
    import pytest
    from tarp37.packing import visible_features
    with pytest.raises(ValueError, match='Different physical/risk contract'):
        visible_features(dict(physics_version='native37_complete_plan_ac_hour_start_v4',
                              risk_contract='four_boundary_events_complete_plan_v4'), [0.]*51)


def test_offline_reuse_cannot_enter_v5_training(tmp_path):
    import pytest
    from tarp37.packing import pack_records
    with pytest.raises(ValueError, match='offline label audit'):
        pack_records([dict(training_eligible=False)], tmp_path/'packed')
