import importlib.util
from pathlib import Path


def test_unified_hardware_is_original_and_finite():
    root = Path(__file__).resolve().parents[1]
    assert (root/'src/tarp37/contract.py').exists(), 'single shared physical contract is missing'
    from tarp37.contract import PhysicalContract
    c = PhysicalContract.load()
    assert len(c.case.buses) == 66 and len(c.case.lines) == 64
    assert c.line_limit_ka == .9 and (c.v_min, c.v_max) == (.93, 1.07)
    assert sum(c.case.root_capacity_mw.values()) == 60.
    assert sum(g['pmax_mw'] for g in c.case.generators if g['bus'] not in c.case.root_capacity_mw) == 240.
    assert all(g['qmax_mvar'] == 30 for g in c.case.generators)
    assert c.impedance_scale == 1.
    assert len(c.components) == 2


def test_native_station_capacity_and_action_order():
    from tarp37.contract import PhysicalContract
    c = PhysicalContract.load()
    assert len(c.station_names) == 47 and len(c.device_names) == 51
    assert all(n.startswith('scs_') for n in c.station_names[:37])
    assert all(n.startswith('fcs_') for n in c.station_names[37:])
    assert sum(c.station_slots) == 840
