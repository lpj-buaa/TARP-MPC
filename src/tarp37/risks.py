"""Recomputable physical event labels, with separate unknown masks."""
import numpy as np
from .contract import PhysicalContract,RISK_VERSION,RISK_NAMES


def window_labels(rows,horizon=6,utilization=None,wait_sla=None,contract=None):
    c=contract or PhysicalContract.load()
    utilization=c.renewable_utilization_target if utilization is None else utilization
    wait_sla=c.waiting_sla_minutes if wait_sla is None else wait_sla
    events=[]; known=[]
    for k in range(4):
        complete=len(rows)==horizon; positive=False
        if k in (0,1):
            observed=[r for r in rows if r.get('regulation_known' if k==0 else 'pre_known',False)]
            field='regulated_voltage_excess_pu' if k==0 else 'pre_capacity_ratio_peak'
            threshold=c.voltage_tolerance_pu if k==0 else 1+c.capacity_ratio_tolerance
            positive=any(r[field]>threshold for r in observed); complete &= len(observed)==horizon
        elif k==2:
            observed=[r for r in rows if r.get('service_known',False)]
            positive=any(r['departure_energy_deficit_kwh']>c.service_energy_tolerance_kwh or r['fcs_wait_max_minutes']>wait_sla+1e-6 for r in observed)
            complete &= len(observed)==horizon
        else:
            complete &= all(all(f'{s}_{f}_mwh' in r for s in ['wind','solar'] for f in ['available','curtail']) for r in rows)
            if complete:
                for source in ['wind','solar']:
                    available=sum(r[f'{source}_available_mwh'] for r in rows)
                    curtailed=sum(r[f'{source}_curtail_mwh'] for r in rows)
                    positive |= available>c.renewable_energy_tolerance_mwh and curtailed>(1-utilization)*available+c.renewable_energy_tolerance_mwh
        events.append(float(positive)); known.append(bool(positive or complete))
    return np.array(events,np.float32),np.array(known,bool)


def source_curtailment_ratios(row,contract=None):
    c=contract or PhysicalContract.load()
    return {s:row.get(s+'_curtail_mwh',0.)/row.get(s+'_available_mwh',1.)
        if row.get(s+'_available_mwh',0.)>c.renewable_energy_tolerance_mwh else 0.
        for s in ('wind','solar')}


def intensities(row,contract=None):
    c=contract or PhysicalContract.load()
    voltage=row.get('regulated_voltage_excess_pu') if row.get('regulation_known') else None
    capacity=row.get('pre_capacity_ratio_peak')
    return [voltage/.01 if voltage is not None else None,
            max(capacity-1.,0.)/.1 if capacity is not None else None,
            max(row.get('departure_energy_deficit_kwh',0.)/10.,row.get('fcs_wait_max_minutes',0.)/c.waiting_sla_minutes),
            max(source_curtailment_ratios(row,c).values())]


def hourly_events(row,contract=None):
    """Only true event boundaries; the RE budget requires the complete window."""
    c=contract or PhysicalContract.load()
    return dict(Voltage=bool(row['regulated_voltage_excess_pu']>c.voltage_tolerance_pu) if row.get('regulation_known') else None,
        NetworkCapacity=bool(row['pre_capacity_ratio_peak']>1+c.capacity_ratio_tolerance) if row['pre_known'] else None,
        ChargingService=bool(row['departure_energy_deficit_kwh']>c.service_energy_tolerance_kwh or
            row['fcs_wait_max_minutes']>c.waiting_sla_minutes+1e-6) if row['service_known'] else None,
        RenewableCurtailment=None)
