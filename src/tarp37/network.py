"""One hardware model for MPC and independently checked hourly AC execution."""
import math
import numpy as np
import pandapower as pp
from .contract import PhysicalContract, VERSION
from .assets import activate

activate()
from fpowerkit import Grid


class UnifiedNetwork:
    def __init__(self, contract=None):
        self.c = contract or PhysicalContract.load(); self.case = self.c.case
        self.bus_ids = list(self.case.buses); self.ix = {b:i for i,b in enumerate(self.bus_ids)}
        self.gen = self.case.generators; self.roots = list(self.case.root_capacity_mw)
        self.device_map = np.zeros((66,51))
        for j,b in enumerate(self.c.device_bus): self.device_map[self.ix[b],j] = 1.
        mean = np.array([np.mean([self.case.base_load(h*3600)[b][0] for h in range(24)]) for b in self.bus_ids])
        qmean = np.array([np.mean([self.case.base_load(h*3600)[b][1] for h in range(24)]) for b in self.bus_ids])
        self.load_share = mean/mean.sum(); self.q_ratio = np.divide(qmean,mean,out=np.zeros(66),where=mean>0)
        self.solar_share = self.load_share.copy()
        self.solar_share[[self.ix[b] for b in self.roots]] = 0.; self.solar_share /= self.solar_share.sum()
        self.wind_share = np.zeros(66)
        for g in self.gen:
            if g['bus'] not in self.roots: self.wind_share[self.ix[g['bus']]] += 1.
        self.wind_share /= self.wind_share.sum()
        self.gen_map = np.zeros((66,10))
        for j,g in enumerate(self.gen): self.gen_map[self.ix[g['bus']],j] = 1.
        self.backup_map = np.zeros((66,2))
        for j,b in enumerate(['b0_18','b1_18']): self.backup_map[self.ix[b],j] = 1.
        self.component_map = np.array([[float(b in comp) for b in self.bus_ids] for comp in self.c.components])
        self.downstream = np.zeros((64,66)); adjacency = {b:[] for b in self.bus_ids}
        for j,l in enumerate(self.case.lines):
            adjacency[l['from_bus']].append((l['to_bus'],j)); adjacency[l['to_bus']].append((l['from_bus'],j))
        for root in self.roots:
            todo, parent = [root], {root:None}
            for b in todo:
                for nxt, line in adjacency[b]:
                    if nxt not in parent: parent[nxt]=(b,line); todo.append(nxt)
            sub = {b:{b} for b in todo}
            for b in reversed(todo[1:]):
                p,k=parent[b]; self.downstream[k,[self.ix[x] for x in sub[b]]] = 1.; sub[p] |= sub[b]
        self.r_voltage = self.downstream.T@np.diag([l['r_ohm'] for l in self.case.lines])@self.downstream/self.case.base_kv**2
        self.x_voltage = self.downstream.T@np.diag([l['x_ohm'] for l in self.case.lines])@self.downstream/self.case.base_kv**2
        native = Grid.fromFileXML(str(self.case.grid_file))
        self.costs = {g.ID:(g.CostA(0)/native.Sb_MVA**2,g.CostB(0)/native.Sb_MVA,g.CostC(0)) for g in native.Gens}
        self.calls = 0

    def injections(self, exo, power, base_fraction=1.):
        base = self.load_share*float(exo[0])*base_fraction
        p = base+self.device_map@np.asarray(power)-self.wind_share*exo[1]-self.solar_share*exo[2]
        return p, base*self.q_ratio

    def _build(self, exo, power, base_fraction, backup_remaining, constrained=True, local_p=None):
        c = self.c; net = pp.create_empty_network(sn_mva=self.case.base_mva)
        for b in self.bus_ids:
            root = b in self.roots
            pp.create_bus(net,vn_kv=self.case.base_kv,name=b,
                min_vm_pu=1. if root else (c.v_min if constrained else .5),
                max_vm_pu=1. if root else (c.v_max if constrained else 1.5))
        for l in self.case.lines:
            pp.create_line_from_parameters(net,self.ix[l['from_bus']],self.ix[l['to_bus']],1.,l['r_ohm'],l['x_ohm'],0.,
                max_i_ka=c.line_limit_ka,max_loading_percent=100. if constrained else 1e8,name=l['id'])
        base=self.load_share*exo[0]*base_fraction; q=base*self.q_ratio
        dev=self.device_map@np.asarray(power)
        for i in range(66): pp.create_load(net,i,p_mw=base[i]+dev[i],q_mvar=q[i])
        conventional=[]
        for j,g in enumerate(self.gen):
            bounds=dict(min_p_mw=g['pmin_mw'],max_p_mw=g['pmax_mw'],min_q_mvar=g['qmin_mvar'],max_q_mvar=g['qmax_mvar'])
            if g['bus'] in self.roots:
                kind='ext_grid'; ident=pp.create_ext_grid(net,self.ix[g['bus']],vm_pu=1.,name=g['id'],**bounds)
            elif constrained:
                kind='sgen'; ident=pp.create_sgen(net,self.ix[g['bus']],0.,0.,controllable=True,name=g['id'],**bounds)
            else:
                kind='gen'; ident=pp.create_gen(net,self.ix[g['bus']],p_mw=float(local_p[j]),vm_pu=1.,name=g['id'],**bounds)
            if constrained:
                a,b,z=self.costs[g['id']]
                factor=max(float(exo[3]),1.)/30. if kind=='ext_grid' else 1.
                pp.create_poly_cost(net,ident,kind,cp0_eur=z,cp1_eur_per_mw=b*factor,cp2_eur_per_mw2=a*factor)
            conventional.append((g,kind,ident))
        renew=[]
        for source,shares,available in [('wind',self.wind_share,exo[1]),('solar',self.solar_share,exo[2])]:
            for bus in np.flatnonzero(shares>0):
                cap=float(shares[bus]*available)
                if cap<=1e-10: continue
                i=pp.create_sgen(net,bus,p_mw=0. if constrained else cap,q_mvar=0.,controllable=constrained,
                    min_p_mw=0.,max_p_mw=cap,min_q_mvar=0.,max_q_mvar=0.,name=source+'_'+self.bus_ids[bus])
                if constrained: pp.create_poly_cost(net,i,'sgen',cp1_eur_per_mw=-200.)
                renew.append((source,int(bus),i,cap))
        backups=[]
        if constrained:
            for j in range(2):
                bus=int(np.argmax(self.backup_map[:,j])); cap=min(c.backup_power_mw,max(float(backup_remaining[j]),0.))
                i=pp.create_sgen(net,bus,p_mw=0.,q_mvar=0.,controllable=True,min_p_mw=0.,max_p_mw=cap,
                    min_q_mvar=0.,max_q_mvar=0.,name='finite_backup_'+str(j))
                pp.create_poly_cost(net,i,'sgen',cp1_eur_per_mw=5000.); backups.append((bus,i))
        return net,conventional,renew,backups,base,q,dev

    def _read(self, net, conventional, renew, backups, base, q, dev):
        voltage=net.res_bus.vm_pu.to_numpy(float)
        vc=voltage*np.exp(1j*np.deg2rad(net.res_bus.va_degree.to_numpy(float)))
        inj=-base-dev-1j*q; gens=[]
        for g,kind,i in conventional:
            row=net['res_'+kind].loc[i]; p=float(row.p_mw); qq=float(row.q_mvar)
            gens.append(dict(id=g['id'],bus=g['bus'],p_mw=p,q_mvar=qq)); inj[self.ix[g['bus']]]+=complex(p,qq)
        used={'wind':0.,'solar':0.}; rs=[]; renewable_by_bus=np.zeros(66)
        for source,bus,i,cap in renew:
            value=float(net.res_sgen.at[i,'p_mw']); used[source]+=value; inj[bus]+=value; renewable_by_bus[bus]+=value
            rs.append(dict(source=source,bus=self.bus_ids[bus],used_mw=value,available_mw=cap))
        backup=[]
        for bus,i in backups:
            value=float(net.res_sgen.at[i,'p_mw']); backup.append(value); inj[bus]+=value
        y=np.zeros((66,66),complex); current=[]; loss=0.
        for l in self.case.lines:
            i,j=self.ix[l['from_bus']],self.ix[l['to_bus']]
            z=complex(l['r_ohm'],l['x_ohm'])*self.case.base_mva/self.case.base_kv**2
            yy=1/z; y[i,i]+=yy; y[j,j]+=yy; y[i,j]-=yy; y[j,i]-=yy
            amp=abs(vc[i]-vc[j])*self.case.base_kv/(math.sqrt(3)*abs(complex(l['r_ohm'],l['x_ohm'])))
            current.append(float(amp)); loss+=3*amp**2*l['r_ohm']
        residual=float(np.max(np.abs(vc*np.conj(y@vc)*self.case.base_mva-inj)))
        root=[next(g['p_mw'] for g in gens if g['bus']==b) for b in self.roots]
        gex=max(max(s['pmin_mw']-g['p_mw'],g['p_mw']-s['pmax_mw'],s['qmin_mvar']-g['q_mvar'],g['q_mvar']-s['qmax_mvar'],0.)
                  for s,g in zip(self.gen,gens))
        vex=max(self.c.v_min-float(voltage.min()),float(voltage.max())-self.c.v_max,0.)
        ratio=np.array(current)/self.c.line_limit_ka
        capacity=max(float(ratio.max()),max(max(p/30.,1-p/30.) for p in root))
        rootgen=sum(root); local=sum(g['p_mw'] for g in gens)-rootgen
        return dict(converged=True,voltage=voltage.tolist(),current_ka=current,loading_ratio=ratio.tolist(),
            max_line_loading_ratio=float(ratio.max()),max_bus_residual_mva=residual,generator_bound_excess=gex,
            voltage_excess_pu=float(vex),capacity_ratio_peak=float(capacity),feeder_import_mw=root,
            grid_import_mw=float(rootgen),local_generation_mw=float(local),generators=gens,
            renewable_generators=rs,wind_used_mw=used['wind'],solar_used_mw=used['solar'],
            backup_by_feeder_mw=backup or [0.,0.],backup_mwh=float(sum(backup)),loss_mwh=float(loss),
            physical_valid=bool(vex<=1e-6 and ratio.max()<=1+1e-6 and gex<=1e-6 and residual<=1e-5),
            net_p=(base+dev-renewable_by_bus-self.backup_map@np.array(backup or [0.,0.])).tolist(),q_mvar=q.tolist(),
            renewable_used_by_bus_mw=renewable_by_bus.tolist())

    def dispatch(self, exo, power, backup_remaining, allow_shedding=True):
        """Finite proportional non-EV shedding ladder; EV delivery is held fixed.

        Each trial is stateless. Only the chosen result is committed by environment.
        Local P/Q and renewable utilization are optimized within the same contract.
        """
        self.calls+=1; exo=np.asarray(exo,float); power=np.asarray(power,float)
        if exo.shape!=(4,) or power.shape!=(51,) or not np.isfinite(np.r_[exo,power]).all(): raise ValueError('Invalid AC inputs')
        attempts=[]
        for fraction in ([1.,.9,.75,.5,.25,0.] if allow_shedding else [1.]):
            net,gen,re,back,base,q,dev=self._build(exo,power,fraction,backup_remaining)
            try:
                pp.runopp(net,verbose=False,init='flat',numba=False,delta=1e-7,OPF_VIOLATION=1e-7,OPF_FLOW_LIM=2)
                # Commit finite, physically bounded setpoints, then rebalance the
                # original AC equations. OPF zero-width variables can otherwise
                # contain a small negative numerical backup/renewable output.
                for idx, spec in net.sgen.iterrows():
                    net.sgen.at[idx,'p_mw'] = float(np.clip(net.res_sgen.at[idx,'p_mw'],spec.min_p_mw,spec.max_p_mw))
                    net.sgen.at[idx,'q_mvar'] = float(np.clip(net.res_sgen.at[idx,'q_mvar'],spec.min_q_mvar,spec.max_q_mvar))
                pp.runpp(net,init='results',numba=False,max_iteration=80,tolerance_mva=1e-10,calculate_voltage_angles=True)
                result=self._read(net,gen,re,back,base,q,dev)
            except (pp.OPFNotConverged,pp.LoadflowNotConverged):
                attempts.append(dict(base_fraction=fraction,status='numerical_unknown')); continue
            attempts.append(dict(base_fraction=fraction,status='valid' if result['physical_valid'] else 'independent_check_failed'))
            if not result['physical_valid']: continue
            shed=exo[0]*(1-fraction); wc=max(exo[1]-result['wind_used_mw'],0.); sc=max(exo[2]-result['solar_used_mw'],0.)
            cost=0.
            for g in result['generators']:
                a,b,z=self.costs[g['id']]; factor=max(float(exo[3]),1.)/30. if g['bus'] in self.roots else 1.
                cost+=factor*(a*g['p_mw']**2+b*g['p_mw']+z)
            recovery=shed*100000.+(wc+sc)*200.+result['backup_mwh']*5000.
            result.update(status='executed_ac_valid',physics_version=VERSION,non_ev_shed_mwh=float(shed),
                non_ev_shed_by_bus_mw=(self.load_share*shed).tolist(),base_fraction=fraction,
                wind_curtail_mwh=float(wc),solar_curtail_mwh=float(sc),
                wind_available_mwh=float(exo[1]),solar_available_mwh=float(exo[2]),
                generation_cost=float(cost),recovery_cost=float(recovery),total_operating_cost=float(cost+recovery),
                protection_attempts=attempts,protection_policy='AC_dispatch_then_finite_proportional_nonEV_shedding')
            return result
        return dict(physical_valid=False,converged=False,status='no_verified_execution_point',protection_attempts=attempts)

    def pressure(self, exo, power, planned_generation=None):
        """Unexecuted request test: native bounded local PV regulation, roots slack.

        Roots may violate their declared P/Q limits in this diagnostic solve;
        those exceedances are measured and never accepted as actual operation.
        """
        local=np.zeros(10) if planned_generation is None else np.asarray(planned_generation,float)
        net,gen,re,back,base,q,dev=self._build(np.asarray(exo,float),power,1.,[0.,0.],False,local)
        try:
            pp.runpp(net,init='flat',numba=False,enforce_q_lims=True,max_iteration=80,tolerance_mva=1e-9,calculate_voltage_angles=True)
            r=self._read(net,gen,re,back,base,q,dev); r['scope']='request_only_not_executed'; return r
        except pp.LoadflowNotConverged:
            return dict(converged=False,scope='request_numerical_unknown_not_a_failure_label')
