"""Recomputable physical event labels, with separate unknown masks."""
import numpy as np
from .contract import RISK_VERSION,RISK_NAMES


def window_labels(rows,horizon=6,utilization=.9,wait_sla=30.):
    events=[]; known=[]
    for k in range(4):
        complete=len(rows)==horizon; positive=False
        if k in (0,1):
            observed=[r for r in rows if r.get('pre_known',False)]
            field='pre_voltage_excess_pu' if k==0 else 'pre_capacity_ratio_peak'
            threshold=1e-6 if k==0 else 1+1e-6
            positive=any(r[field]>threshold for r in observed); complete &= len(observed)==horizon
        elif k==2:
            observed=[r for r in rows if r.get('service_known',False)]
            positive=any(r['departure_energy_deficit_kwh']>.001 or r['fcs_wait_max_minutes']>wait_sla+1e-6 for r in observed)
            complete &= len(observed)==horizon
        else:
            complete &= all(all(f'{s}_{f}_mwh' in r for s in ['wind','solar'] for f in ['available','curtail']) for r in rows)
            if complete:
                for source in ['wind','solar']:
                    available=sum(r[f'{source}_available_mwh'] for r in rows)
                    curtailed=sum(r[f'{source}_curtail_mwh'] for r in rows)
                    positive |= available>1e-8 and curtailed>(1-utilization)*available+1e-8
        events.append(float(positive)); known.append(bool(positive or complete))
    return np.array(events,np.float32),np.array(known,bool)


def intensities(row):
    voltage=row.get('pre_voltage_excess_pu')
    capacity=row.get('pre_capacity_ratio_peak')
    available=row.get('wind_available_mwh',0.)+row.get('solar_available_mwh',0.)
    curtailed=row.get('wind_curtail_mwh',0.)+row.get('solar_curtail_mwh',0.)
    return [voltage/.01 if voltage is not None else None,
            max(capacity-1.,0.)/.1 if capacity is not None else None,
            max(row.get('departure_energy_deficit_kwh',0.)/10.,row.get('fcs_wait_max_minutes',0.)/30.),
            curtailed/available if available>1e-8 else 0.]
