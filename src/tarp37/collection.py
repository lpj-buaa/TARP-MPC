from pathlib import Path
from datetime import date
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import hashlib
import json
import time
import numpy as np
from scipy.stats import qmc
from .assets import PROJECT,WORKSPACE,TRAFFIC_SOURCE,sha,frozen_inputs
from .contract import PhysicalContract,VERSION,RISK_VERSION
from .risks import window_labels


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda v:v.tolist() if isinstance(v,np.ndarray) else int(v) if isinstance(v,np.integer) else float(v)),encoding='utf-8')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(manifest):
    for p,h in {**manifest['inputs'],**manifest['outputs']}.items():
        if not Path(p).is_file() or sha(p)!=h: raise ValueError('Frozen input or output changed: '+p)


def stamp(folder,inputs,**extra):
    folder=Path(folder)
    outputs={str(p):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='complete.json'}
    write(folder/'complete.json',dict(inputs=inputs,outputs=outputs,**extra))


def validate_sources(sources):
    for i,s in enumerate(sources):
        for other in sources[:i]:
            delta=abs((date.fromisoformat(s['date'])-date.fromisoformat(other['date'])).days)
            if delta==0 or (s['role']!=other['role'] and delta<4): raise ValueError('Source date overlap across full history/branch/forecast windows')


def initialize(root,config,inputs,resume=False):
    root=Path(root); marker=root/'audit/complete.json'
    if root.exists() and not resume: raise ValueError('RunId already exists; use -Resume or a new RunId')
    if marker.exists():
        saved=read(marker)
        if read(root/'audit/config.json')!=config or saved['inputs']!=inputs: raise ValueError('Configuration or code changed; use a new RunId')
        verify(saved); return True
    (root/'audit').mkdir(parents=True,exist_ok=True); write(root/'audit/config.json',config)
    write(root/'audit/contract.json',PhysicalContract.load().manifest())
    from .data import ForecastTrace
    write(root/'audit/renewable_scenario.json',ForecastTrace().reference)
    from .network import UnifiedNetwork
    write(root/'audit/operating_economics.json',UnifiedNetwork().economics())
    stamp(root/'audit',inputs,rl_training_calls=0,old_expert_training_calls=0)
    return False


def simulation_inputs():
    files=['assets.py','contract.py','data.py','network.py','mpc.py','traffic.py','environment.py','risks.py','evidence.py','collection.py']
    inputs={str(PROJECT/'src/tarp37'/p):sha(PROJECT/'src/tarp37'/p) for p in files}
    inputs.update(frozen_inputs())
    for p in (TRAFFIC_SOURCE/'case').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc': inputs[str(p)]=sha(p)
    verify(read(TRAFFIC_SOURCE/'collection_complete.json'))
    for p in (PROJECT/'artifacts/data').glob('*'): inputs[str(p)]=sha(p)
    for p in [WORKSPACE/'tarp_v2sim_traffic_diagnosis/adapter.py',
              WORKSPACE/'tarp_v2sim_engineering_grid/artifacts/normal_ev_5000_s42_v1/profile.json',
              WORKSPACE/'tarp_v2sim_engineering_grid/artifacts/normal_ev_5000_s42_v1/collection_complete.json']:
        inputs[str(p)]=sha(p)
    return inputs


def candidate_actions(reference,view,contract,count,seed):
    reference=np.asarray(reference,float)
    options=[('mpc_reference',reference.copy())]
    low=reference.copy(); low[:47]*=.2; options.append(('charging_throttle',low))
    release=reference.copy(); release[47:]=-np.asarray(view['discharge_cap_mw'])[47:]
    release[:37]=-np.minimum(view['discharge_cap_mw'][:37],view['v2g_available_mwh'][:37])
    options.append(('release_reserve',release))
    maximum=reference.copy(); maximum[:47]=contract.station_power_envelope_mw; options.append(('maximum_native_rate_envelope',maximum))
    if count>4:
        points=qmc.Sobol(51,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(count-4))))[:count-4]
        for j,p in enumerate(points):
            a=np.r_[contract.station_power_envelope_mw,np.ones(4)]*p
            a[47:]=np.where(p[47:]>.5,(p[47:]-.5)*2*view['charge_cap_mw'][47:],(p[47:]-.5)*2*view['discharge_cap_mw'][47:])
            a[:37]=np.where(p[:37]>.2,a[:37],-view['discharge_cap_mw'][:37]*p[:37]*5)
            options.append((f'sobol_{j:03}',a))
    return options[:count]


def source_job(root,config,source):
    from .environment import UnifiedEnvironment,ExecutionUnknown
    root=Path(root); sid=source['date']+'_s'+str(source.get('seed',42)); out=root/'sources'/sid
    if (out/'complete.json').exists(): verify(read(out/'complete.json')); return sid
    if out.exists(): raise ValueError('Incomplete native source retained; choose a new RunId: '+sid)
    out.mkdir(parents=True)
    env=UnifiedEnvironment(out/'simulation',source['date'],source.get('seed',42),config.get('v2g_participation',.5))
    warmup=[]; records=[]; first_hour={}; checks=0; started=time.perf_counter()
    profiles=config.get('preference_anchors',[[.25]*4,[.05,.85,.05,.05]])
    try:
        for hour in sorted(config['branch_hours']):
            while env.hour<hour:
                plan=env.plan()
                row=env.step(plan['power_mw'],plan)
                warmup.append(dict(hour=row['hour'],physical_valid=row['physical_valid'],
                    v_min=min(row['voltage']),v_max=max(row['voltage']),line_ratio=row['max_line_loading_ratio'],
                    local_generation_mw=row['local_generation_mw'],grid_import_mw=row['grid_import_mw'],
                    non_ev_shed_mwh=row['non_ev_shed_mwh'],backup_mwh=row['backup_mwh']))
                if env.hour%4==0: print(f'{sid}: warmup {env.hour}/{hour}h',flush=True)
            snapshot=env.snapshot(out/f'pool_{hour}')
            plan=env.plan()
            candidates=candidate_actions(plan['power_mw'],env.view(),env.c,config['candidate_count'],config.get('candidate_seed',42)+hour)
            for p,anchor in enumerate(profiles):
                for a,(family,power) in enumerate(candidates):
                    env.restore(snapshot); env.anchor=np.asarray(anchor,float)
                    context=env.context(); context['candidate_mw']=np.asarray(power).tolist()
                    rows=[]
                    for h in range(6):
                        current=plan if h==0 else env.plan()
                        try: rows.append(env.step(power if h==0 else current['power_mw'],current))
                        except ExecutionUnknown as error:
                            rows.append(error.row); break
                    first=np.array(rows[0]['actual_device_mw']+rows[0]['ess_energy_mwh']+rows[0].get('backup_energy_remaining_mwh',[]))
                    if a in first_hour:
                        if not np.allclose(first_hour[a],first,atol=1e-6): raise RuntimeError('Same first action changed under a future-only preference')
                        checks+=1
                    else: first_hour[a]=first
                    events,known=window_labels(rows,contract=env.c)
                    records.append(dict(source_id=sid,role=source['role'],date=source['date'],pool_hour=hour,
                        profile_id=p,candidate_id=a,family=family,context=context,rows=rows,event=events.tolist(),event_known=known.tolist()))
                    print(f'{sid}: pool {hour}, preference {p+1}/{len(profiles)}, action {a+1}/{len(candidates)}: '
                          f'events={events.astype(int).tolist()}, AC={sum(r["physical_valid"] for r in rows)}/{len(rows)}',flush=True)
            env.restore(snapshot); first_hour={}
    finally: env.close()
    inputs=read(root/'audit/complete.json')['inputs']
    verify(dict(inputs=inputs,outputs={}))
    write(out/'warmup.json',warmup); write(out/'records.json',records)
    # Publish compact records, not disposable native diagnostics/snapshots, as
    # stage outputs. Native files remain available for targeted reproducibility.
    write(out/'complete.json',dict(inputs=inputs,outputs={str(out/n):sha(out/n) for n in ['warmup.json','records.json']},
        first_hour_checks=checks,records=len(records),elapsed_s=time.perf_counter()-started,
        rl_training_calls=0,expert_training_calls=0,physics_version=VERSION,risk_contract=RISK_VERSION))
    return sid


def analyze(root):
    from .traffic import aggregate_departure_gaps
    root=Path(root); records=[]; warm=[]; checks=0
    for marker in sorted((root/'sources').glob('*/complete.json')):
        m=read(marker); verify(m); checks+=m['first_hour_checks']
        records+=read(marker.parent/'records.json'); warm+=read(marker.parent/'warmup.json')
    all_rows=[r for rec in records for r in rec['rows']]; rows=[r for r in all_rows if r['physical_valid']]
    summary=dict(sources=len(list((root/'sources').glob('*/complete.json'))),branches=len(records),future_hours=len(all_rows),
        physical_valid_hours=len(rows),unknown_hours=len(all_rows)-len(rows),first_hour_preference_invariance_checks=checks,
        max_voltage_pu=max((max(r['voltage']) for r in rows),default=None),min_voltage_pu=min((min(r['voltage']) for r in rows),default=None),
        max_line_loading_ratio=max((r['max_line_loading_ratio'] for r in rows),default=None),
        max_ac_residual_mva=max((r['max_bus_residual_mva'] for r in rows),default=None),
        max_battery_balance_error_kwh=max((r['battery_balance_error_kwh'] for r in rows),default=None),
        max_ess_balance_error_mwh=max((r['ess_balance_error_mwh'] for r in rows),default=None),
        min_backup_remaining_mwh=min((min(r['backup_energy_remaining_mwh']) for r in rows),default=None),
        allocation_guard_adjustments=sum(len(r['protection_trials'])>1 for r in rows),
        risk_counts=np.sum([r['event'] for r in records],axis=0).astype(int).tolist(),
        compound_branches=sum(sum(r['event'])>=2 for r in records),
        requested_line_exceedance_hours=sum(max(r.get('pre_line_loading_ratio') or [0.])>1+1e-6 for r in rows),
        pre_routine_AC_known_hours=sum(r.get('pre_known',False) for r in rows),
        pre_routine_voltage_event_hours=sum(r.get('risk_events_hour',{}).get('Voltage') is True for r in rows),
        pre_routine_capacity_event_hours=sum(r.get('risk_events_hour',{}).get('NetworkCapacity') is True for r in rows),
        pre_export_permission_excess_hours=sum((r.get('pre_pcc_export_permission_excess_mw') or 0.)>1e-6 for r in rows),
        numerical_shedding_fallback_hours=sum(r.get('correction_evidence',{}).get('numerical_fallback_used',False) and r['non_ev_shed_mwh']>1e-6 for r in rows),
        warmup_ac_valid=sum(r['physical_valid'] for r in warm),warmup_hours=len(warm),
        rl_training_calls=0,old_expert_training_calls=0,physics_version=VERSION,risk_contract=RISK_VERSION)
    paired=[]
    references={(r['source_id'],r['pool_hour'],r['profile_id']):r for r in records if r['candidate_id']==0}
    for r in records:
        ref=references[(r['source_id'],r['pool_hour'],r['profile_id'])]
        if r['candidate_id']==0: continue
        if len(r['rows'])!=6 or len(ref['rows'])!=6 or not all(x['physical_valid'] for x in r['rows']+ref['rows']): continue
        fields=['total_operating_cost','loss_mwh','queue_vehicle_min','departure_energy_deficit_kwh','wind_curtail_mwh','solar_curtail_mwh']
        def total(rec,field):
            if field=='departure_energy_deficit_kwh':
                return aggregate_departure_gaps([e for row in rec['rows'] for e in row['traffic']['service_events']])
            return sum(x[field] for x in rec['rows'])
        paired.append(dict(source_id=r['source_id'],pool_hour=r['pool_hour'],profile_id=r['profile_id'],family=r['family'],
            differences={f:total(r,f)-total(ref,f) for f in fields},
            terminal_ess_delta_mwh=(np.array(r['rows'][-1]['ess_energy_mwh'])-ref['rows'][-1]['ess_energy_mwh']).tolist(),
        future_control_l1_mwh=float(np.abs(np.array([x['actual_device_mw'] for x in r['rows'][1:]])-
                                             [x['actual_device_mw'] for x in ref['rows'][1:]]).sum())))
    out=root/'analysis'; out.mkdir(exist_ok=True); write(out/'summary.json',summary); write(out/'paired_changes.json',paired)
    text='# 37nodes统一闭环小规模核验\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n'+json.dumps(paired,ensure_ascii=False,indent=2)
    (out/'RESULTS_CN.md').write_text(text,encoding='utf-8')
    return summary


def run_collection(config,run_id,resume=False,workers=1,until='analyze'):
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',run_id): raise ValueError('Invalid RunId')
    validate_sources(config['sources']); root=PROJECT/'artifacts/runs'/run_id; inputs=simulation_inputs()
    initialize(root,config,inputs,resume); c=PhysicalContract.load()
    contract_path=root/'audit/contract.json'
    if not contract_path.exists(): write(contract_path,c.manifest())
    if until=='audit': return dict(root=str(root),stage='audit')
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            tasks=[pool.submit(source_job,root,config,s) for s in config['sources']]
            for task in as_completed(tasks): task.result()
    else:
        for source in config['sources']: source_job(root,config,source)
    if until=='collect': return dict(root=str(root),stage='collect')
    summary=analyze(root); verify(dict(inputs=inputs,outputs={}))
    return dict(root=str(root),stage='analyze',summary=summary)
