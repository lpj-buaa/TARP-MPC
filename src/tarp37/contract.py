from dataclasses import dataclass, field
from pathlib import Path
import gzip
import json
import xml.etree.ElementTree as ET
import numpy as np
from .assets import activate, CASE, TRAFFIC_SOURCE, sha

activate()
from native_grid.case import CaseInventory

VERSION = 'native37_complete_plan_ac_hour_start_v4'
RISK_VERSION = 'four_boundary_events_complete_plan_v4'
RISK_NAMES = ['Voltage', 'NetworkCapacity', 'ChargingService', 'RenewableCurtailment']


@dataclass
class PhysicalContract:
    case: object
    station_names: list
    station_slots: list
    station_power_envelope_mw: np.ndarray
    device_names: list
    device_bus: list
    components: list
    line_limit_ka: float = .9
    v_min: float = .93
    v_max: float = 1.07
    root_voltage_pu: float = 1.
    impedance_scale: float = 1.
    backup_power_mw: float = 1.
    backup_initial_mwh: float = 6.
    waiting_sla_minutes: float = 30.
    renewable_utilization_target: float = .9
    voltage_tolerance_pu: float = 1e-6
    capacity_ratio_tolerance: float = 1e-6
    power_tolerance_mw: float = 1e-6
    service_energy_tolerance_kwh: float = .001
    renewable_energy_tolerance_mwh: float = 1e-8
    ess_capacity_mwh: float = 2.
    ess_min_mwh: float = .2
    ess_max_mwh: float = 1.8
    ess_power_mw: float = 1.
    ess_efficiency: float = .95

    @classmethod
    def load(cls):
        case = CaseInventory.read(CASE)
        files = [next(CASE.glob('*.scs.xml')), next(CASE.glob('*.fcs.xml'))]
        nodes = [s for p in files for s in ET.parse(p).getroot()]
        # Original XML does not specify a finite per-pile kW rating. The native
        # implementation uses infinity and each connected vehicle's own limit.
        # This planning envelope is therefore an upper bound from native EV types,
        # NOT an invented charger nameplate.
        max_power = {'pcs': 0., 'pcf': 0.}
        vehicle_file = next((TRAFFIC_SOURCE/'case').glob('*.veh.xml.gz'))
        with gzip.open(vehicle_file, 'rt', encoding='utf-8') as handle:
            for _, e in ET.iterparse(handle, events=['end']):
                if e.tag == 'ev':
                    for k in max_power: max_power[k] = max(max_power[k], float(e.get(k)))
                    e.clear()
        slots = [int(s.get('slots')) for s in nodes]
        cap = [n*min(float(s.get('max_pc', 'inf')), max_power['pcs' if s.tag == 'scs' else 'pcf'])/1000
               for s, n in zip(nodes, slots)]
        adjacency = {b: set() for b in case.buses}
        for line in case.lines:
            adjacency[line['from_bus']].add(line['to_bus']); adjacency[line['to_bus']].add(line['from_bus'])
        remaining, components = set(case.buses), []
        while remaining:
            todo = [min(remaining)]; seen = set(todo)
            for b in todo:
                for nxt in sorted(adjacency[b] - seen): seen.add(nxt); todo.append(nxt)
            remaining -= seen; components.append(sorted(seen))
        names = [s.get('name') for s in nodes]
        ess_buses = ['b0_8', 'b0_24', 'b1_8', 'b1_24']
        return cls(case, names, slots, np.array(cap), names+['ESS_'+b for b in ess_buses],
                   [s.get('bus') for s in nodes]+ess_buses, components)

    def manifest(self):
        return dict(version=VERSION, risk_version=RISK_VERSION, risk_names=RISK_NAMES,
            electrical_risk_scope='complete_routine_plan_at_realized_delivery_before_extra_AC_recourse',
            raw_request_diagnostic='separate forecast-based allocation pressure, not a realized physical event',
            case_dir=str(CASE), grid_sha256=sha(self.case.grid_file),
            bus_order=list(self.case.buses), device_order=self.device_names, device_bus=self.device_bus,
            line_limit_ka=self.line_limit_ka, voltage_limits_pu=[self.v_min, self.v_max],
            impedance_scale=self.impedance_scale, root_capacity_mw=self.case.root_capacity_mw,
            generators=self.case.generators, components=self.components,
            station_slots=self.station_slots, station_power_envelope_mw=self.station_power_envelope_mw.tolist(),
            station_power_semantics='Original unspecified pile limits; bound is slots times maximum native EV rate',
            added_ess=dict(count=4, capacity_mwh=2., power_mw=1., eta=.95, bounds_mwh=[.2,1.8]),
            emergency=dict(power_mw_per_feeder=self.backup_power_mw, initial_energy_mwh_per_feeder=self.backup_initial_mwh),
            clock=dict(grid_seconds=3600, traffic_statistics_seconds=900, exchange_seconds=10, traffic_physics_seconds=1),
            generation_model='Native independent P/Q bounds; root Pmin also retained; no implicit export allowance',
            curtailment_target=self.renewable_utilization_target, fcs_wait_sla_minutes=self.waiting_sla_minutes,
            numerical_tolerances=dict(voltage_pu=self.voltage_tolerance_pu,capacity_ratio=self.capacity_ratio_tolerance,
                power_mw=self.power_tolerance_mw,service_energy_kwh=self.service_energy_tolerance_kwh,
                renewable_energy_mwh=self.renewable_energy_tolerance_mwh),
            risk_semantics=dict(Voltage='routine AC voltage outside limits, not deviation or instability',
                NetworkCapacity='line ampacity or positive PCC import exceeds capacity; export permission recorded separately',
                ChargingService='native mobility/recovery energy deficit or start-of-charge waiting exceeds scenario SLA',
                RenewableCurtailment='either wind or solar exceeds the six-hour operating curtailment budget, not statutory annual compliance'))
