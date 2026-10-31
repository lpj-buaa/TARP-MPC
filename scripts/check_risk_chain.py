"""Independent saved-record audit; never fits models or changes source results."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tarp37.assets import PROJECT
from tarp37.contract import PhysicalContract,RISK_VERSION
from tarp37.network import UnifiedNetwork
from tarp37.data import ForecastTrace
from tarp37.collection import read,write,verify


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-id',required=True); args=p.parse_args()
    root=PROJECT/'artifacts/runs'/args.run_id
    c=PhysicalContract.load(); n=UnifiedNetwork(c); trace=ForecastTrace(); records=[]
    for marker in sorted((root/'sources').glob('*/complete.json')):
        verify(read(marker)); records.extend(read(marker.parent/'records.json'))
    if not records: raise ValueError('No completed source records')
    errors=[]; checks=0; pre_known=0; quality_only=0; max_q_diff=0.; max_re_diff=0.
    voltage_hours=0; capacity_hours=0; export_hours=0; fallback_shed_hours=0
    post_min=[]; post_max=[]; post_loading=[]; routine_q=[]; controls=[]
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok: errors.append(message)
    for rec in records:
        for h,r in enumerate(rec['rows']):
            prefix=f"{rec['source_id']}/{rec['candidate_id']}/h{h}"
            check(r['risk_contract']==RISK_VERSION,prefix+': stale risk contract')
            if not r['physical_valid']: continue
            v=np.array(r['voltage']); current=np.array(r['current_ka'])
            check(v.min()>=c.v_min-c.voltage_tolerance_pu and v.max()<=c.v_max+c.voltage_tolerance_pu,prefix+': post voltage')
            check(current.max()<=c.line_limit_ka*(1+c.capacity_ratio_tolerance),prefix+': post ampacity')
            check(r['max_bus_residual_mva']<=1e-5,prefix+': post AC balance')
            delivered=np.array(r['actual_device_mw']); applied=np.array(r['applied_allocation_mw'])
            check(np.allclose(delivered[:47],r['traffic']['station_mean_mw'],atol=1e-10),prefix+': changed EV delivery')
            check(np.all(np.maximum(delivered[:47],0.)<=np.maximum(applied[:47],0.)+1e-8),prefix+': exceeded charging allocation')
            check(r['battery_balance_error_kwh']<=1e-5 and r['ess_balance_error_mwh']<=1e-8,prefix+': resource balance')
            guard=r['forecast_allocation_guard']; point=r['routine_operating_point']; pre=r['pre_routine_AC']
            check(guard['available_before_delivery'],prefix+': noncausal guard')
            check(np.allclose(applied,np.array(r['bounded_request_mw'])*guard['allocation_scale']),prefix+': allocation scale')
            check(np.allclose(guard['forecast_values'],trace.visible(r['now'])['values'][0]),prefix+': guard forecast mismatch')
            exo=trace.actual(r['now'])
            check(np.isclose(r['non_ev_shed_mwh'],exo[0]*(1-r['base_fraction']),atol=1e-8),prefix+': base shedding energy')
            check(np.isclose(r['backup_mwh'],sum(r['backup_by_feeder_mw']),atol=1e-8),prefix+': backup energy')
            for source,k in [('wind',1),('solar',2)]:
                check(np.isclose(r[source+'_curtail_mwh'],max(exo[k]-r[source+'_used_mw'],0.),atol=1e-8),prefix+': '+source+' curtailment')
            post_min.append(float(v.min())); post_max.append(float(v.max())); post_loading.append(float(current.max()/c.line_limit_ka))
            routine_q.append(float(np.abs(point['q_mvar']).sum()))
            e=r['correction_evidence']
            fallback_shed_hours+=bool(e['numerical_fallback_used'] and r['non_ev_shed_mwh']>1e-6)
            check(not e['shedding_necessity_proven'],prefix+': unsupported unavoidable shedding claim')
            if not pre['known']:
                check(not r['pre_known'] and not e['correction_comparison_known'],prefix+': unknown AC made known')
                continue
            pre_known+=1
            pv=np.array(pre['voltage']); signed=np.minimum(pv-c.v_min,c.v_max-pv)
            ve=float(np.maximum(-signed,0.).max())
            ratio=max(float(max(pre['loading_ratio'])),max(max(p,0.)/c.case.root_capacity_mw[b]
                for p,b in zip(pre['feeder_import_mw'],n.roots)))
            check(np.isclose(r['pre_voltage_excess_pu'],ve,atol=1e-12),prefix+': voltage event quantity')
            check(np.isclose(r['pre_capacity_ratio_peak'],ratio,atol=1e-12),prefix+': thermal/import quantity')
            check(r['risk_events_hour']['Voltage']==(ve>c.voltage_tolerance_pu),prefix+': voltage quality made event')
            check(r['risk_events_hour']['NetworkCapacity']==(ratio>1+c.capacity_ratio_tolerance),prefix+': capacity event')
            voltage_hours+=ve>c.voltage_tolerance_pu; capacity_hours+=ratio>1+c.capacity_ratio_tolerance
            export_hours+=pre['pcc_export_permission_excess_mw']>c.power_tolerance_mw
            quality_only+=bool(np.max(abs(pv-1.))>1e-4 and ve<=c.voltage_tolerance_pu)
            for j,g in enumerate(pre['generators']):
                if g['bus'] in n.roots: continue
                diff=abs(g['q_mvar']-point['q_mvar'][j]); max_q_diff=max(max_q_diff,diff)
                check(diff<=1e-8 and abs(g['p_mw']-point['generation_mw'][j])<=1e-8,prefix+': changed routine local P/Q')
            for gen in pre['renewable_generators']:
                want=min(gen['available_mw'],point['renewable_used_by_source_bus_mw'][gen['source']][n.ix[gen['bus']]])
                diff=abs(gen['used_mw']-want); max_re_diff=max(max_re_diff,diff)
                check(diff<=1e-8,prefix+': restored curtailed renewable injection')
            check(np.allclose(pre['backup_by_feeder_mw'],point['backup_mw'],atol=1e-9),prefix+': changed routine backup')
            check(pre['base_fraction']==point['base_fraction'],prefix+': ignored routine base shedding')
            check(np.isclose(e['base_shedding_change_mwh'],exo[0]*(point['base_fraction']-r['base_fraction']),atol=1e-8),prefix+': additional shedding attribution')
            controls.append(dict(source=rec['source_id'],candidate=rec['candidate_id'],hour=h+1,
                triggers=e['triggered_constraints'],voltage_excess_pu=ve,capacity_ratio=ratio,
                guard_scale=guard['allocation_scale'],extra_shed_mwh=e['additional_non_ev_shed_mwh'],
                extra_backup_mwh=e['additional_backup_mwh'],wind_used_change_mwh=e['wind_used_change_mwh'],
                solar_used_change_mwh=e['solar_used_change_mwh']))
    summary=dict(checks=checks,errors=errors,branches=len(records),future_hours=sum(len(x['rows']) for x in records),
        pre_AC_known_hours=pre_known,pre_voltage_event_hours=int(voltage_hours),pre_capacity_event_hours=int(capacity_hours),
        separate_export_permission_excess_hours=int(export_hours),quality_deviation_without_voltage_event_hours=int(quality_only),
        max_fixed_local_q_difference_mvar=max_q_diff,max_fixed_RE_command_difference_mw=max_re_diff,
        post_voltage_range_pu=[min(post_min),max(post_max)],post_max_line_loading_ratio=max(post_loading),
        routine_sum_abs_Q_mvar_range=[min(routine_q),max(routine_q)],numerical_fallback_shedding_hours=fallback_shed_hours,
        window_risk_counts=np.sum([r['event'] for r in records],axis=0).astype(int).tolist(),
        model_training_calls=0,risk_contract=RISK_VERSION,
        counting_unit='branch-hour records; matched branches share initial state and are not independent samples')
    write(root/'analysis/risk_chain_audit.json',summary)
    write(root/'analysis/correction_records.json',controls)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(1)


if __name__=='__main__': main()
