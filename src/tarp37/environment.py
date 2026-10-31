from pathlib import Path
from copy import deepcopy
import json
import pickle
import shutil
import numpy as np
from .assets import TRAFFIC_SOURCE
from .network import UnifiedNetwork
from .traffic import NativeTraffic
from .data import ForecastTrace
from .mpc import BusinessMPC,update_preferences,routine_point
from .contract import RISK_VERSION,VERSION
from .risks import intensities,hourly_events,source_curtailment_ratios
from .evidence import correction_evidence,constraint_triggers


class ExecutionUnknown(RuntimeError):
    def __init__(self,row):
        self.row=row
        super().__init__('Post-delivery AC recourse unverified; delivered native traffic is retained')


class UnifiedEnvironment:
    def __init__(self,output,date='2024-04-01',seed=42,participation=.5):
        self.output=Path(output); self.output.mkdir(parents=True,exist_ok=True)
        self.network=UnifiedNetwork(); self.c=self.network.c; self.trace=ForecastTrace()
        self.epoch=self.trace.time_for_date(date); self.hour=0; self.date=date; self.seed=seed
        self.traffic=NativeTraffic(self.output/'traffic',participation,seed)
        self.controller=BusinessMPC(self.network); self.ess=np.ones(4); self.backup=np.full(2,self.c.backup_initial_mwh)
        self.weights=np.full(4,.25); self.anchor=np.full(4,.25); self.previous_power=np.zeros(51); self.history=[]; self._plan_cache=None
        self.last_grid=self.network.dispatch(self.trace.actual(self.epoch-3600),np.zeros(51),self.backup)
        if not self.last_grid['physical_valid']: raise RuntimeError('Initial grid has no verified physical observation')
        profile=json.loads((TRAFFIC_SOURCE/'profile.json').read_text(encoding='utf-8'))
        self.prior=np.zeros((24,47))
        for r in profile['rows']:
            h=int(r['time_s'])//3600%24
            self.prior[h]+=np.array([r['station_power_mw'].get(n,0.) for n in self.c.station_names])/4
        # Existing source profile contains 48 retained hours, averaged by day.
        days=len(profile['rows'])/96
        if days>1: self.prior/=days

    @property
    def now(self): return self.epoch+self.hour*3600
    def forecast(self): return self.trace.visible(self.now)

    def plan(self):
        key=(self.hour,tuple(self.weights),tuple(self.ess),tuple(self.backup),tuple(self.previous_power))
        if self._plan_cache is None or self._plan_cache[0]!=key:
            self._plan_cache=(key,self.controller.plan(self.view(),self.forecast(),self.weights,self.backup))
        return deepcopy(self._plan_cache[1])

    def view(self):
        v=self.traffic.view(); v['ess_energy_mwh']=self.ess.copy(); v['previous_power_mw']=self.previous_power.copy()
        v['arrival_prior_mw']=np.array([self.prior[(self.hour+k)%24] for k in range(6)])
        v['charge_cap_mw'][47:]=np.minimum(self.c.ess_power_mw,np.maximum((self.c.ess_max_mwh-self.ess)/self.c.ess_efficiency,0.))
        v['discharge_cap_mw'][47:]=np.minimum(self.c.ess_power_mw,np.maximum((self.ess-self.c.ess_min_mwh)*self.c.ess_efficiency,0.))
        return v

    def context(self):
        v=self.view(); f=self.forecast()
        return dict(now=self.now,hour=self.hour,date=self.date,physics_version=VERSION,risk_contract=RISK_VERSION,
            view={k:np.asarray(x).tolist() if isinstance(x,np.ndarray) else x for k,x in v.items()},
            visible_forecast=f['values'].tolist(),forecast_issued_at=f['issued_at'].tolist(),weights=self.weights.tolist(),
            preference_anchor=self.anchor.tolist(),backup_energy_mwh=self.backup.tolist(),
            bus_voltage_pu=self.last_grid['voltage'],bus_net_p_mw=self.last_grid['net_p'],bus_q_mvar=self.last_grid['q_mvar'],
            routine_generation_mw=self.plan()['planned_generation_mw'][0],
            routine_q_mvar=self.plan()['planned_q_mvar'][0],
            routine_operating_point=routine_point(self.plan()),history=deepcopy(self.history[-2:]))

    def _bounded(self,request):
        r=np.asarray(request,float).copy(); v=self.view()
        if r.shape!=(51,) or not np.isfinite(r).all(): raise ValueError('51 finite physical MW allocations required')
        r[:47]=np.minimum(r[:47],self.c.station_power_envelope_mw)
        r[:37]=np.maximum(r[:37],-np.minimum(v['discharge_cap_mw'][:37],v['v2g_available_mwh'][:37]))
        r[37:47]=np.maximum(r[37:47],0.)
        r[47:]=np.clip(r[47:],-v['discharge_cap_mw'][47:],v['charge_cap_mw'][47:])
        return r

    def step(self,request,plan=None):
        request=np.asarray(request,float); bounded=self._bounded(request); now=self.now
        before_ess=self.ess.copy(); before_backup=self.backup.copy(); weights=self.weights.copy()
        # This baseline is part of the visible, deterministic controller state.
        # A caller's optional plan cannot silently change diagnostic truth.
        routine=self.plan(); point=routine_point(routine); planned_pg=np.asarray(point['generation_mw'])
        visible=self.forecast(); predicted=visible['values'][0]
        guard_request=self.network.pressure(predicted,bounded,point)
        attempts=[]
        # Choose allocation BEFORE observing any native arrivals or realized
        # current-hour average. Endpoint tests are an allocation guard, not a
        # proof for every signed partial-delivery realization.
        for scale in [1.,.75,.5,.25,0.]:
            applied=bounded*scale
            zero=applied.copy(); zero[:47]=0.
            high=self.network.dispatch(predicted,applied,before_backup,allow_shedding=False)
            low=self.network.dispatch(predicted,zero,before_backup,allow_shedding=False)
            attempts.append(dict(scale=scale,full_endpoint_valid=high['physical_valid'],zero_EV_endpoint_valid=low['physical_valid'],
                full_endpoint_status=high.get('status'),zero_EV_endpoint_status=low.get('status')))
            if high['physical_valid'] and low['physical_valid']: break
        guard_applied=self.network.pressure(predicted,applied,point)
        tr=self.traffic.advance_hour(applied[:47])
        actual=np.r_[tr['station_mean_mw'],applied[47:]]
        # Actuals are first queried for settlement after irreversible native
        # delivery. Only generation/Q, curtailment and non-EV shedding can adapt.
        exo=self.trace.actual(now); pre=self.network.pressure(exo,actual,point)
        ac=self.network.dispatch(exo,actual,before_backup,allow_shedding=True)
        self.ess=before_ess+np.maximum(actual[47:],0.)*self.c.ess_efficiency-np.maximum(-actual[47:],0.)/self.c.ess_efficiency
        ess_error=np.max(np.abs(self.ess-before_ess-np.maximum(actual[47:],0.)*self.c.ess_efficiency+np.maximum(-actual[47:],0.)/self.c.ess_efficiency))
        if not ac['physical_valid']:
            raise ExecutionUnknown(dict(physical_valid=False,now=now,hour=self.hour,risk_contract=RISK_VERSION,
                traffic=tr,actual_device_mw=actual.tolist(),ess_energy_mwh=self.ess.tolist(),
                service_known=True,departure_energy_deficit_kwh=tr['departure_energy_deficit_kwh'],
                fcs_wait_max_minutes=tr['fcs_wait_max_minutes'],pre_known=pre['known'],
                pre_voltage_excess_pu=pre.get('voltage_excess_pu'),pre_capacity_ratio_peak=pre.get('capacity_ratio_peak'),
                routine_operating_point=point,pre_routine_AC=pre,
                correction_evidence=correction_evidence(pre,ac,exo,self.c),
                execution_status='delivery_retained_AC_unknown_suffix_unknown',protection_trials=attempts))
        self.backup=before_backup-np.array(ac['backup_by_feeder_mw'])
        if self.backup.min() < -1e-7 or self.ess.min()<.2-1e-7 or self.ess.max()>1.8+1e-7: raise RuntimeError('Finite resource violation')
        allocation_cut=float(np.maximum(np.maximum(bounded[:47],0)-np.maximum(applied[:47],0),0).sum())
        row=dict(**ac,hour=self.hour,time_s=self.traffic.inst.ctime-3600,now=now,risk_contract=RISK_VERSION,
            candidate_request_mw=request.tolist(),bounded_request_mw=bounded.tolist(),applied_allocation_mw=applied.tolist(),
            actual_device_mw=actual.tolist(),allocation_capacity_reduction_mw=allocation_cut,
            battery_balance_error_kwh=abs(tr['battery_balance_error_kwh']),ess_balance_error_mwh=float(ess_error),
            backup_energy_remaining_mwh=self.backup.tolist(),ess_energy_mwh=self.ess.tolist(),
            pre_known=pre['known'],pre_voltage_excess_pu=pre.get('voltage_excess_pu'),
            pre_capacity_ratio_peak=pre.get('capacity_ratio_peak'),pre_voltage=pre.get('voltage'),
            pre_line_loading_ratio=pre.get('loading_ratio'),
            pre_pcc_import_ratio_peak=pre.get('pcc_import_ratio_peak'),
            pre_pcc_import_excess_mw=pre.get('pcc_import_excess_mw'),
            pre_pcc_export_permission_excess_mw=pre.get('pcc_export_permission_excess_mw'),
            pre_voltage_margin_pu=pre.get('voltage_margin_pu'),
            pre_minimum_voltage_margin_pu=pre.get('minimum_voltage_margin_pu'),
            pre_scope='complete_routine_plan_at_realized_delivery_before_extra_AC_recourse',
            service_known=True,departure_energy_deficit_kwh=tr['departure_energy_deficit_kwh'],
            fcs_wait_max_minutes=tr['fcs_wait_max_minutes'],queue_vehicle_min=tr['queue_vehicle_min'],
            native_rescue_count=tr['native_rescue_count'],traffic=tr,weights=weights.tolist(),
            mobility_recovery_energy_deficit_kwh=tr['departure_energy_deficit_kwh'],
            service_deficit_scope='native failed departure: gap to reachable recovery charging, not all contractual charging requests',
            generation_plan_mw=planned_pg.tolist(),protection_trials=attempts,
            routine_operating_point=point,pre_routine_AC=pre,
            correction_evidence=correction_evidence(pre,ac,exo,self.c),
            forecast_allocation_guard=dict(available_before_delivery=True,allocation_scale=scale,
                changed=bool(scale<1.),diagnostic_known=guard_request['known'],
                requested_constraints=constraint_triggers(guard_request,self.c),
                diagnostic=guard_request,applied_diagnostic=guard_applied,forecast_values=predicted.tolist(),
                trigger_scope='forecast diagnostic only; not a realized risk event or proof of per-control necessity'),
            pre_guard_forecast_voltage_excess_pu=guard_request.get('voltage_excess_pu'),
            pre_guard_forecast_capacity_ratio_peak=guard_request.get('capacity_ratio_peak'),
            electrical_risk_scope='complete_routine_plan_at_realized_delivery_before_extra_AC_recourse',
            forecast_issued_at=visible['issued_at'].tolist(),
            protection_information='visible_forecast_before_native_delivery',
            recourse_semantics='hourly_quasi_steady_fixed_EV_delivery')
        row['intensity']=intensities(row,self.c)
        row['risk_events_hour']=hourly_events(row,self.c)
        row['source_curtailment_ratios']=source_curtailment_ratios(row,self.c)
        self.previous_power=actual.copy(); self.last_grid=ac; self.hour+=1
        self.history=(self.history+[dict(intensity=row['intensity'],actual_device_mw=actual.tolist(),
            ess_energy_mwh=self.ess.tolist(),backup_energy_mwh=self.backup.tolist(),weights=weights.tolist(),
            voltage=ac['voltage'],queue_vehicle_min=tr['queue_vehicle_min'],
            traffic_quarters=deepcopy(tr['quarters']),loss_mwh=ac['loss_mwh'],grid_import_mw=ac['grid_import_mw'],
            local_generation_mw=ac['local_generation_mw'],capacity_ratio=ac['max_line_loading_ratio'],
            departure_energy_deficit_kwh=tr['departure_energy_deficit_kwh'])])[-2:]
        v=self.view(); f=self.forecast()['values']
        self.weights=update_preferences(self.weights,self.anchor,
            price_pressure=float(np.clip((f[0,3]-np.mean(f[:,3]))/50,-1,1)),
            service_pressure=float(min(1.,np.sum(v['need_mwh'])/max(np.sum(v['charge_cap_mw'][:47])*.9,1e-6)+v['queue_count']/100)),
            resource_pressure=float(np.clip((4-self.ess.sum())/4,-1,1)),
            change_pressure=float(min(1.,np.abs(actual-bounded).sum()/5)))
        return row

    def snapshot(self,folder):
        folder=Path(folder); self.traffic.snapshot(folder/'traffic')
        state={k:deepcopy(getattr(self,k)) for k in ['hour','ess','backup','weights','anchor','previous_power','history','last_grid']}
        with (folder/'state.pkl').open('wb') as f: pickle.dump(state,f)
        return folder

    def restore(self,folder):
        folder=Path(folder); self.traffic.restore(folder/'traffic')
        with (folder/'state.pkl').open('rb') as f: state=pickle.load(f)
        for k,v in state.items(): setattr(self,k,v)
        self._plan_cache=None

    def close(self): self.traffic.close()
