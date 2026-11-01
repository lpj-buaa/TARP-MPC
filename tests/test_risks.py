import numpy as np


def test_four_risks_share_new_voltage_capacity_and_real_service_thresholds():
    from tarp37.risks import window_labels
    base=dict(pre_voltage_excess_pu=0.,pre_capacity_ratio_peak=.8,pre_known=True,
        regulation_known=True,regulated_voltage_excess_pu=0.,
        departure_energy_deficit_kwh=0.,fcs_wait_max_minutes=0.,service_known=True,
        wind_available_mwh=1.,solar_available_mwh=1.,wind_curtail_mwh=0.,solar_curtail_mwh=0.)
    rows=[dict(base) for _ in range(6)]
    rows[0]['regulated_voltage_excess_pu']=.001; rows[1]['pre_capacity_ratio_peak']=1.01
    rows[2]['fcs_wait_max_minutes']=31.; rows[3]['solar_curtail_mwh']=1.
    labels,known=window_labels(rows)
    assert labels.tolist()==[1,1,1,1] and known.all()


def test_partial_curtailment_ratio_is_unknown_and_positive_stress_survives():
    from tarp37.risks import window_labels
    rows=[dict(pre_voltage_excess_pu=.01,pre_capacity_ratio_peak=.5,pre_known=True,
        regulation_known=True,regulated_voltage_excess_pu=.01,
        service_known=True,departure_energy_deficit_kwh=0,fcs_wait_max_minutes=0,
        wind_available_mwh=1,wind_curtail_mwh=1,solar_available_mwh=0,solar_curtail_mwh=0)]
    labels,known=window_labels(rows)
    assert labels[0]==1 and known[0]
    assert not known[1] and not known[3]
