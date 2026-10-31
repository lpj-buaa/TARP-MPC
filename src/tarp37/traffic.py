"""Native V2SIM actions, exact energy metering, and native snapshot/restore."""
from pathlib import Path
from copy import deepcopy
import hashlib
import math
import pickle
import random
import shutil
import numpy as np
from .assets import activate, TRAFFIC_SOURCE
from .contract import PhysicalContract

activate()
from adapter import configure, native
native()
from v2sim import V2SimInstance,TimeConfig,UXsimConfig,LoadStateOption
from v2sim.sim.ux import TrafficUX
from v2sim.veh import EV,GV,VehStatus
from v2sim.veh.veh import Vehicle

_METER=None


def aggregate_departure_gaps(events):
    peaks={}
    for event in events:
        if event['kind'] not in ('departure_delay','native_rescue'): continue
        key=(event['vehicle'],event['trip_id'])
        peaks[key]=max(peaks.get(key,0.),event['gap_kwh'])
    return float(sum(peaks.values()))


def safe_retry_time(now, energy, required, charging_kwh_s, step_s):
    delay=max(step_s,int(1+max(required-energy,0.)/charging_kwh_s)) if charging_kwh_s>1e-12 else step_s
    return int(now+delay)


def _install_meter():
    for cls,name,kind in [(EV,'charge','charge'),(EV,'_bidirectional_charge','charge'),
                          (EV,'bidirectional_discharge','discharge'),(Vehicle,'drive','trip')]:
        original=getattr(cls,name)
        if getattr(original,'_tarp37_metered',False): continue
        def wrapped(veh,*args,_original=original,_kind=kind,**kwargs):
            before=veh._energy; result=_original(veh,*args,**kwargs)
            if _METER is not None:
                delta=veh._energy-before
                _METER[_kind+'_battery_kwh']+=delta if _kind=='charge' else -delta
                if _kind=='charge' and delta>1e-12 and veh._leave_at_etar: _METER['positive_fcs'].add(veh.name)
            return result
        wrapped._tarp37_metered=True; setattr(cls,name,wrapped)


def grid_v2g_allocator(env, count, grid_request_kwh_s, available):
    """Native discharge setter is battery-side; command and metering are grid-side."""
    vehicles=list(env.EVs)
    caps=np.array([v._pdv*v._ed for v in vehicles])
    remaining=min(max(grid_request_kwh_s,0.),float(caps.sum()))
    alloc=np.zeros(len(vehicles)); active=list(range(len(vehicles)))
    while active and remaining>1e-14:
        each=remaining/len(active); following=[]
        for i in active:
            value=min(each,caps[i]-alloc[i]); alloc[i]+=value; remaining-=value
            if alloc[i]<caps[i]-1e-14: following.append(i)
        active=following
    for v,p in zip(vehicles,alloc): v.set_temp_pd(float(p/max(v._ed,1e-12)))


class SafeTrafficUX(TrafficUX):
    def _TrafficUX__batch_depart(self):
        """Native queue semantics, with zero-charge and past-time retry corrected."""
        while not self._que.empty() and self._que.top[0]<=self._ct:
            depart_time,(veh_id,extras)=self._que.pop(); veh=self._vehs[veh_id]
            if extras is not None:
                trip,force_sc,force_fc,force_fcs=extras
                if isinstance(veh,EV): veh._force_sc=force_sc
                veh._fr_on_dpt=force_fc; veh._dpt_rs=force_fcs
            if self._TrafficUX__start_trip(veh):
                self._log.depart(self._ct,veh,max(0,self._ct-depart_time),veh._cs); continue
            hub=self._hubs.gs if isinstance(veh,GV) else self._hubs.fcs
            names=hub.get_online_names(self._ct)
            if not names: raise RuntimeError('No native recovery station available')
            nodes={}
            for name in names:
                bind=self._hubs.get_bind_of(name)
                if bind not in nodes or self._hubs[nodes[bind]].wait_count()>veh._w*self._hubs[name].wait_count(): nodes[bind]=name
            route=self.find_best_route(veh.trip.O,nodes.keys(),False)
            if not route.nodes: raise RuntimeError('Unreachable native recovery route; no artificial successful arrival')
            station=nodes[route.nodes[-1]]; required=route.length*veh._epm*veh._kr
            if isinstance(veh,EV) and self._hubs.scs.has_veh(veh._name):
                retry=safe_retry_time(self._ct,veh._energy,required,veh._pcr,self._step)
                self._log.depart_delay(self._ct,veh,required,retry-self._ct)
                self._que.push(retry,(veh_id,None))
            else:
                # Preserve, label, and cost native roadside rescue; it is not a
                # successful road-engine arrival and not a new supply generator.
                veh.status=VehStatus.Depleted; veh._cs=station
                retry=int(self._ct+2*route.travelTime); self._fQ.push(retry,veh._name)
                self._log.depart_failed(self._ct,veh,required,station,retry)

    def _prepare_stations(self,veh,to_stations,omega,to_charge,hub):
        nodes,scores={},{}
        for name in to_stations:
            s=hub[name]; bind=s._bind
            # Vehicle/slot constraints and the current announced site allocation
            # determine expected service, rather than a fixed 30-minute proxy.
            if hasattr(s,'_pc_limtot'):
                native_cap=s._slots*veh._pcf
                cap=min(s._pc_limtot,native_cap)
                if cap<=1e-12: minutes=360.
                else:
                    workload=sum(max(v._etar-v._energy,0.)/max(v._ecf,1e-12) for v in s._chi)
                    queued=sum(max(v._etar-v._energy,0.)/max(v._ecf,1e-12) for v in s._buf)
                    wait=(workload+queued)/cap/60 if len(s._chi)>=s._slots else 0.
                    rate=min(veh._pcf,s._pc_lim1,cap/max(min(s._slots,len(s._chi)+1),1))
                    minutes=wait+to_charge/max(rate,1e-12)/60
                score=s.pbuy(self._ct,veh)*to_charge+omega*minutes
            else: score=s.pbuy(self._ct,veh)*to_charge+omega*(s.wait_count()+1)*5.
            if bind not in nodes or score<scores[bind]: nodes[bind]=name; scores[bind]=score
        return nodes,scores


class NativeTraffic:
    def __init__(self, output, participation=.5, seed=42):
        _install_meter(); self.output=Path(output); self.output.mkdir(parents=True,exist_ok=True)
        self.case=self.output/'case'; shutil.copytree(TRAFFIC_SOURCE/'case',self.case)
        self.c=PhysicalContract.load(); self.seed=seed; self.participation=participation; self.restore_count=0
        self.waiting={}; self.deadlines=set(); self.events=[]; self.counts={}; self.history=[]
        self.inst=self._new(self.output/'initial')
        configure(self.inst,seed,1.,True)
        for v in self.inst.vehicles.evs.values():
            opt=int(hashlib.sha256(v.name.encode()).hexdigest()[:8],16)/2**32 < participation
            v._kv=.7 if opt else 1.0001
        # Explicit aggregator participation contract; does not alter road or
        # charger hardware and is separate from a new retail tariff mechanism.
        for s in self.inst.scs:
            s._psell_is_serv_fee=False
            s._pd_alloc=grid_v2g_allocator
        self._attach(); self.inst.start()

    def _new(self, log, state=None):
        args=dict(out_dir=str(log),seed=self.seed,silent=True,config=UXsimConfig(no_parallel=True),
                  disabled_plugins=['pdn','v2g'],logging_items=[])
        if state is not None: args.update(state_option=LoadStateOption.FromGiven,state_dir=str(state))
        result=V2SimInstance.from_project(str(self.case),TimeConfig(0,10,4*86400),**args)
        result.core.__class__=SafeTrafficUX
        return result

    def _attach(self):
        log=self.inst.core.trip_logger
        def fcs(t,v,cs,dist): self.waiting[v.name]=t
        def delay(t,v,required,delay): self.events.append(dict(kind='departure_delay',time_s=t,vehicle=v.name,trip_id=v.trip_id,gap_kwh=max(required-v._energy,0.)))
        def failed(t,v,required,cs,retry): self.events.append(dict(kind='native_rescue',time_s=t,vehicle=v.name,trip_id=v.trip_id,gap_kwh=max(required-v._energy,0.)))
        log.add_arrive_fcs_listener(fcs); log.add_depart_delay_listener(delay); log.add_depart_failed_listener(failed)
        original=self.inst.core.W.get_arrived_vehicles
        def checked():
            for name,obj in original():
                if obj.state!='end': raise RuntimeError('Aborted native road movement cannot be a successful arrival')
                yield name,obj
        self.inst.core.W.get_arrived_vehicles=checked

    def view(self):
        stations={s.name:s for s in [*self.inst.scs,*self.inst.fcs]}
        caps=[]; discharge=[]; need=[]; due=[]; surplus=[]
        for name in self.c.station_names:
            s=stations[name]; fast=name.startswith('fcs_'); evs=list(s._chi)
            caps.append(sum(min(v._pcf if fast else v._pcs,s._pc_lim1) for v in evs)*3.6)
            need.append(sum(max(v._etar-v._energy,0.) for v in [*evs,*s._buf])/1000)
            due.append(min([max((v.trip.depart_time-self.inst.ctime)/3600,1.) for v in evs] or [6.]) if not fast else 1.)
            eligible=[v for v in evs if not fast and v._kv<1 and v.soc>v._kv]
            surplus.append(sum(max(v._energy-v._kv*v._cap,0.)*v._ed for v in eligible)/1000)
            discharge.append(sum(v._pdv*v._ed for v in eligible)*3.6)
        return dict(charge_cap_mw=np.r_[caps,np.ones(4)],discharge_cap_mw=np.r_[discharge,np.ones(4)],
                    need_mwh=np.array(need),due_hour=np.array(due),v2g_available_mwh=np.array(surplus),
                    queue_count=sum(s.wait_count() for s in stations.values()),
                    vehicle_energy_kwh=sum(v._energy for v in self.inst.vehicles.evs.values()))

    def advance_hour(self, request):
        global _METER
        request=np.asarray(request,float)
        if request.shape!=(47,) or not np.isfinite(request).all() or np.any(request[37:]<0): raise ValueError('Invalid native station action')
        stations={s.name:s for s in [*self.inst.scs,*self.inst.fcs]}; ordered=[stations[n] for n in self.c.station_names]
        before=sum(v._energy for v in self.inst.vehicles.evs.values()); self.events=[]
        _METER=dict(charge_battery_kwh=0.,discharge_battery_kwh=0.,trip_battery_kwh=0.,positive_fcs=set())
        quarters=[]; energy=np.zeros(47); start=self.inst.ctime; waitmax=0.; queue_minutes=0.
        try:
            for q in range(4):
                qenergy=np.zeros(47)
                for _ in range(90):
                    for s,p in zip(ordered,request): s.set_Pc_lim(max(p,0.)/3.6)
                    # Capacity list refresh uses the actual next station-update clock.
                    self.inst.scs.get_V2G_cap(self.inst.ctime+10)
                    self.inst.scs.set_V2G_demand([max(-p,0.)/3.6 for p in request[:37]])
                    self.inst.step()
                    if self.inst.core.W.world.TIME!=self.inst.ctime: raise RuntimeError('Native clock drift')
                    p=np.array([s.Pc_MW-s.Pd_MW for s in ordered]); qenergy+=p*10/3600
                    for name in list(_METER['positive_fcs']):
                        if name in self.waiting:
                            waitmax=max(waitmax,(self.inst.ctime-self.waiting.pop(name))/60)
                    _METER['positive_fcs'].clear()
                    if self.waiting: waitmax=max(waitmax,max((self.inst.ctime-t)/60 for t in self.waiting.values()))
                    queue_minutes+=sum(s.wait_count() for s in ordered)*10/60
                energy+=qenergy; quarters.append(dict(time_s=start+q*900,station_mean_mw=(qenergy/.25).tolist()))
            after=sum(v._energy for v in self.inst.vehicles.evs.values())
            error=after-before-_METER['charge_battery_kwh']+_METER['discharge_battery_kwh']+_METER['trip_battery_kwh']
            result=dict(time_s=start,station_mean_mw=energy.tolist(),quarters=quarters,battery_balance_error_kwh=float(error),
                charging_battery_kwh=_METER['charge_battery_kwh'],discharging_battery_kwh=_METER['discharge_battery_kwh'],
                trip_energy_kwh=_METER['trip_battery_kwh'],queue_vehicle_min=float(queue_minutes),
                fcs_wait_max_minutes=float(waitmax),departure_energy_deficit_kwh=aggregate_departure_gaps(self.events),
                deficit_semantics='sum of vehicle-trip peak gaps within hour; deduplicate trips across a window',
                service_events=deepcopy(self.events),native_rescue_count=sum(e['kind']=='native_rescue' for e in self.events))
            if abs(error)>1e-5: raise RuntimeError('Vehicle battery conservation failed')
            return result
        finally: _METER=None

    def snapshot(self, folder):
        folder=Path(folder); self.inst.save(folder)
        state=dict(waiting=self.waiting,deadlines=self.deadlines,python_rng=random.getstate(),numpy_rng=np.random.get_state())
        with (folder/'observer.pkl').open('wb') as handle: pickle.dump(state,handle)
        return folder

    def restore(self, folder):
        self.inst.stop(); self.restore_count+=1
        self.inst=self._new(self.output/f'restore_{self.restore_count}',Path(folder))
        with (Path(folder)/'observer.pkl').open('rb') as handle: state=pickle.load(handle)
        self.waiting=state['waiting']; self.deadlines=state['deadlines']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        # Do not install a second checked-arrival closure: native state already
        # retains the first instance's single-world arrival queue wrapper.
        self._attach(); self.inst.start()

    def close(self): self.inst.stop()
