"""Directional resource projections and endogenous controller response.

These descriptors are not the volume of the AC/service feasible region.
The physical constants below are the unchanged shared contract.
"""
import numpy as np

GROUPS=(108,15,20)


def describe(record, station_envelope):
    rows=record['rows']
    if len(rows)!=6 or not all(r.get('physical_valid') for r in rows):return None
    capabilities=[]; responses=[]; weights=[]
    scales=np.array([max(np.sum(station_envelope[:37]),1e-6),max(np.sum(station_envelope[37:]),1e-6),4.])
    for h,r in enumerate(rows):
        if any(k not in r for k in ['ess_energy_mwh','backup_energy_remaining_mwh','candidate_request_mw','weights']):return None
        e=np.asarray(r['ess_energy_mwh']); back=np.asarray(r['backup_energy_remaining_mwh'])
        if e.shape!=(4,) or back.shape!=(2,) or np.any(e<.2-1e-6) or np.any(e>1.8+1e-6):return None
        discharge=np.maximum(e-.2,0.)*.95
        absorb=np.maximum(1.8-e,0.)/.95
        capabilities.extend(np.r_[discharge/(1.6*.95),absorb/(1.6/.95),
                                  np.minimum(discharge,1.),np.minimum(absorb,1.),back/6.])
        if h:
            u=np.asarray(r['candidate_request_mw'])
            if u.shape!=(51,):return None
            responses.extend(np.array([u[:37].sum(),u[37:47].sum(),u[47:].sum()])/scales)
            weights.extend(r['weights'])
    z=np.r_[np.asarray(capabilities)/np.sqrt(GROUPS[0]),
            np.asarray(responses)/np.sqrt(GROUPS[1]),np.asarray(weights)/np.sqrt(GROUPS[2])]/np.sqrt(3.)
    return z if np.isfinite(z).all() else None


def visible_state(context):
    """Only information available before requesting the intervention."""
    v=context['view']; f=np.asarray(context['visible_forecast'])
    result=np.r_[v['ess_energy_mwh'],context['backup_energy_mwh'],context['weights'],context['preference_anchor'],
        np.sum(v['need_mwh']),v['queue_count'],
        np.sum(v['charge_cap_mw'][:37]),np.sum(v['charge_cap_mw'][37:47]),
        np.sum(v['discharge_cap_mw'][:37]),np.sum(v['v2g_available_mwh']),
        np.quantile(v['due_hour'],[0,.5,1]),f.flatten(),np.sin(context['hour']*np.pi/12),np.cos(context['hour']*np.pi/12)]
    if not np.isfinite(result).all():raise ValueError('Nonfinite public state')
    return result


def context_key(r):return f"{r['source_id']}|{r['pool_hour']}|{r['profile_id']}"
