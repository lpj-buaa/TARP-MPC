import numpy as np
from tarp_boundary.learning import knowledge_features

SPATIAL_GROUPS=(66,66,66,51,51,47,47,47,51,282)
PHYSICAL_DIM=49+sum(SPATIAL_GROUPS)
MEMORY_DIM=18+3*85+88


def spatial_raw(ctx):
    v=ctx['view']
    groups=[ctx[k] for k in ['bus_voltage_pu','bus_net_p_mw','bus_q_mvar']]
    groups += [v[k] for k in ['charge_cap_mw','discharge_cap_mw','need_mwh','due_hour',
                              'v2g_available_mwh','previous_power_mw','arrival_prior_mw']]
    flat=np.concatenate([np.asarray(g,float).ravel() for g in groups])
    assert flat.shape==(PHYSICAL_DIM-49,) and np.all(np.isfinite(flat))
    return flat


def state_input(x,quantitative):
    return np.r_[np.asarray(x['state'])/np.sqrt(49),x['spatial'] if quantitative else np.zeros(PHYSICAL_DIM-49)]


def memory(x,observed,quantitative):
    old=np.asarray(knowledge_features(x,observed))
    if not quantitative:return np.r_[old,np.zeros(MEMORY_DIM-18)]
    known=[r for r in observed if r['key']==x['key'] and r['id']!=x['id']]
    family=[r for r in known if r['family'] in [x['family'],'reference']]
    lower=max((r for r in family if r['dose']<=x['dose']),key=lambda r:r['dose'],default=None)
    upper=min((r for r in family if r['dose']>=x['dose']),key=lambda r:r['dose'],default=None)
    distance=lambda r:np.linalg.norm(np.array(x['action'])-r['action'])/np.sqrt(51)
    nearest=min(known,key=lambda r:(distance(r),r['id']),default=None)
    slots=[]
    for r in [lower,upper,nearest]:
        if r is None:slots.extend(np.zeros(85));continue
        slots.extend(np.r_[1.,r['dose']-x['dose'],distance(r),np.asarray(r['response'])/np.sqrt(60),
                           np.asarray(r['descriptor'])/np.sqrt(22)])
    if known:
        d=np.array([r['descriptor'] for r in known])
        moments=np.r_[d.mean(0),d.std(0),d.min(0),d.max(0)]/np.sqrt(22)
    else:moments=np.zeros(88)
    value=np.r_[old,slots,moments]
    assert len(value)==MEMORY_DIM and np.all(np.isfinite(value))
    return value
