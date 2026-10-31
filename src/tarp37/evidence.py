"""Link measured constraint triggers, corrective controls and executed outcomes."""
import numpy as np
from .contract import PhysicalContract


def constraint_triggers(pre,contract=None):
    c=contract or PhysicalContract.load()
    if not pre.get('known',False): return []
    flags=dict(voltage=pre['voltage_excess_pu']>c.voltage_tolerance_pu,
        network_capacity=pre['capacity_ratio_peak']>1+c.capacity_ratio_tolerance,
        export_permission=pre.get('pcc_export_permission_excess_mw',0.)>c.power_tolerance_mw,
        generator_capability=pre.get('generator_bound_excess',0.)>c.power_tolerance_mw)
    return [k for k,v in flags.items() if v]


def correction_evidence(pre,post,exo,contract=None):
    c=contract or PhysicalContract.load()
    known=bool(pre.get('known',False) and post.get('physical_valid',False))
    numerical=any(a['status']=='numerical_unknown' for a in post.get('protection_attempts',[]))
    result=dict(correction_comparison_known=known,actual_AC_valid=bool(post.get('physical_valid',False)),
        triggered_constraints=constraint_triggers(pre,c),numerical_fallback_used=numerical,
        shedding_necessity_proven=False,
        attribution='joint AC redispatch under all bounds; per-control necessity not identified')
    if not known: return result
    pg=np.array([g['p_mw'] for g in pre['generators']]); qg=np.array([g['q_mvar'] for g in pre['generators']])
    result.update(generation_change_mw=(np.array([g['p_mw'] for g in post['generators']])-pg).tolist(),
        q_change_mvar=(np.array([g['q_mvar'] for g in post['generators']])-qg).tolist(),
        wind_used_change_mwh=post['wind_used_mw']-pre['wind_used_mw'],
        solar_used_change_mwh=post['solar_used_mw']-pre['solar_used_mw'],
        backup_change_mwh=(np.array(post['backup_by_feeder_mw'])-pre['backup_by_feeder_mw']).tolist(),
        additional_backup_mwh=max(sum(post['backup_by_feeder_mw'])-sum(pre['backup_by_feeder_mw']),0.),
        base_shedding_change_mwh=float(exo[0])*(pre['base_fraction']-post['base_fraction']),
        additional_non_ev_shed_mwh=max(float(exo[0])*(pre['base_fraction']-post['base_fraction']),0.))
    return result
