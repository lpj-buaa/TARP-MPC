"""Six-hour finite-information business MPC; no predicted-risk objective."""
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.special import softmax
from .contract import VERSION


def routine_point(plan, hour=0):
    """Complete commanded operating point, before additional AC correction."""
    return dict(generation_mw=plan['planned_generation_mw'][hour],
        q_mvar=plan['planned_q_mvar'][hour],
        renewable_used_by_source_bus_mw={
            'wind': plan['planned_wind_used_by_bus_mw'][hour],
            'solar': plan['planned_solar_used_by_bus_mw'][hour]},
        backup_mw=plan['planned_backup_mw'][hour],
        base_fraction=plan['planned_base_fraction'][hour])


def update_preferences(previous, anchor, price_pressure=0., service_pressure=0., resource_pressure=0., change_pressure=0.):
    z=np.log(np.maximum(anchor,.02))+np.clip([price_pressure,service_pressure,resource_pressure,change_pressure],-1,1)
    target=.02+.92*softmax(z)
    return .75*np.asarray(previous)+.25*target


class LP:
    def __init__(self): self.cost=[]; self.bounds=[]; self.eq=[]; self.eqb=[]; self.le=[]; self.leb=[]
    def var(self, lower=0., upper=None, cost=0.):
        i=len(self.cost); self.cost.append(float(cost)); self.bounds.append((float(lower),None if upper is None else float(upper))); return i
    def inequality(self, row, rhs): self.le.append({int(i):float(v) for i,v in row.items() if abs(v)>1e-13}); self.leb.append(float(rhs))
    def equal(self, row, rhs): self.eq.append({int(i):float(v) for i,v in row.items() if abs(v)>1e-13}); self.eqb.append(float(rhs))
    def dense_row(self, indices, coeff): return {int(i):float(v) for i,v in zip(indices,coeff) if abs(v)>1e-13}
    def absolute(self, row, rhs=0., cost=0.):
        i=self.var(cost=cost)
        self.inequality({**row,i:-1.},rhs); self.inequality({**{k:-v for k,v in row.items()},i:-1.},-rhs)
        return i
    def solve(self):
        def matrix(rows):
            rr=[]; cc=[]; vv=[]
            for r,row in enumerate(rows):
                for c,v in row.items(): rr.append(r); cc.append(c); vv.append(v)
            return coo_matrix((vv,(rr,cc)),shape=(len(rows),len(self.cost))).tocsr()
        return linprog(self.cost,A_ub=matrix(self.le),b_ub=self.leb,A_eq=matrix(self.eq),b_eq=self.eqb,bounds=self.bounds,method='highs')


class BusinessMPC:
    def __init__(self, network, horizon=6): self.n=network; self.c=network.c; self.horizon=horizon; self.calls=0

    def _mode(self, view, forecast, weights, backup_remaining, mode):
        n=self.n; c=self.c; h=len(forecast['values']); lp=LP(); w=np.asarray(weights,float)
        power=[]; pg=[]; qg=[]; backups=[]; voltage_rows=[]; essence=[]; curtailments=[]; sheds=[]
        planned_cost_indices=[]; service_indices=[]; resource_indices=[]; change_indices=[]
        prices=forecast['values'][:,3]
        for t,exo in enumerate(forecast['values']):
            charge=np.array(view['charge_cap_mw'],float); discharge=np.array(view['discharge_cap_mw'],float)
            charge[:47]=np.maximum(charge[:47],np.asarray(view['arrival_prior_mw'])[t])
            charge[:47]=np.minimum(charge[:47],c.station_power_envelope_mw)
            signs=np.ones(51)
            if mode=='release' or (mode=='price' and exo[3]>np.median(prices)+1e-6): signs[47:]=-1
            if mode=='price' and exo[3]>np.median(prices)+1e-6: signs[:37]=-1
            power.append(np.array([lp.var(0.,charge[d]) if signs[d]>0 else lp.var(-discharge[d],0.) for d in range(51)]))
            pg.append(np.array([lp.var(g['pmin_mw'],g['pmax_mw']) for g in n.gen]))
            qg.append(np.array([lp.var(g['qmin_mvar'],g['qmax_mvar']) for g in n.gen]))
            backup=np.array([lp.var(0.,min(c.backup_power_mw,float(backup_remaining[j])),cost=n.backup_unit_cost/n.cost_scale) for j in range(2)])
            backups.append(backup)
            for j in range(2): lp.inequality({int(x[j]):1. for x in backups},float(backup_remaining[j]))
            load=n.load_share*exo[0]; renew=n.wind_share*exo[1]+n.solar_share*exo[2]
            shed=lp.var(0.,1.,cost=exo[0]*n.shed_unit_cost/n.cost_scale)
            curt=np.array([lp.var(0.,r,cost=n.curtail_unit_cost/n.cost_scale) for r in renew])
            curtailments.append(curt); sheds.append(shed)
            all_i=np.r_[power[-1],pg[-1],qg[-1],backup,[shed],curt]
            pmap=np.column_stack([n.device_map,-n.gen_map,np.zeros((66,10)),-n.backup_map,-load[:,None],np.eye(66)])
            qmap=np.column_stack([np.zeros((66,51+10)),-n.gen_map,np.zeros((66,2)),-(load*n.q_ratio)[:,None],np.zeros((66,66))])
            bp,bq=load-renew,load*n.q_ratio
            for comp in n.component_map:
                lp.equal(lp.dense_row(all_i,comp@pmap),-float(comp@bp))
                lp.equal(lp.dense_row(all_i,comp@qmap),-float(comp@bq))
            dv=-(n.r_voltage@pmap+n.x_voltage@qmap); v0=1.-n.r_voltage@bp-n.x_voltage@bq
            voltage_rows.append((all_i.copy(),dv,v0))
            for i in range(66):
                lp.inequality(lp.dense_row(all_i,dv[i]),c.v_max-v0[i])
                lp.inequality(lp.dense_row(all_i,-dv[i]),v0[i]-c.v_min)
            pl,ql=n.downstream@bp,n.downstream@bq; pm,qm=n.downstream@pmap,n.downstream@qmap
            # Inscribed 16-sided approximation of the nominal current circle
            # at the common minimum voltage. Actual AC current is checked later.
            rating=math.sqrt(3)*n.case.base_kv*c.line_limit_ka*c.v_min*math.cos(math.pi/16)
            for angle in np.arange(16)*2*math.pi/16:
                co,si=math.cos(angle),math.sin(angle)
                for k in range(64): lp.inequality(lp.dense_row(all_i,co*pm[k]+si*qm[k]),rating-co*pl[k]-si*ql[k])
            for j,g in enumerate(n.gen):
                a,b,z=n.costs[g['id']]; factor=max(float(exo[3]),1.)/30. if g['bus'] in n.roots else 1.
                cuts=np.linspace(g['pmin_mw'],g['pmax_mw'],61)
                segments=[]
                for left,right in zip(cuts[:-1],cuts[1:]):
                    idx=lp.var(0.,right-left,cost=w[0]*factor*(b+a*(left+right))/n.cost_scale)
                    segments.append(idx); planned_cost_indices.append(idx)
                lp.equal({int(pg[-1][j]):1.,**{i:-1. for i in segments}},g['pmin_mw'])
            energies=[]
            for d in range(4):
                e=lp.var(c.ess_min_mwh,c.ess_max_mwh); energies.append(e)
                row={e:1.,int(power[-1][47+d]):-(c.ess_efficiency if signs[47+d]>0 else 1/c.ess_efficiency)}
                if t: row[int(essence[-1][d])]=-1.
                lp.equal(row,float(view['ess_energy_mwh'][d]) if t==0 else 0.)
            essence.append(np.array(energies))
            for d in range(51):
                row={int(power[-1][d]):1.}
                if t: row[int(power[-2][d])]=-1.
                change_indices.append(lp.absolute(row,float(view['previous_power_mw'][d]) if t==0 else 0.,cost=w[3]/5))
            # The same stationary EVs cannot discharge their current surplus twice.
            for d in range(37):
                released={int(power[k][d]):-1. for k in range(t+1) if lp.bounds[int(power[k][d])][0]<0}
                if released: lp.inequality(released,float(view.get('v2g_available_mwh',np.zeros(47))[d]))
        for d in range(47):
            due=min(max(int(view['due_hour'][d]),1),h)
            # Known connected-session requests plus a frozen development arrival
            # prior; never future observed trips or future realized service.
            target=float(view['need_mwh'][d]) + float(np.sum(np.asarray(view['arrival_prior_mw'])[:due,d])*.9)
            shortage=lp.var(cost=w[1]/.1); service_indices.append(shortage)
            row={int(power[t][d]):-(.9 if lp.bounds[int(power[t][d])][0]>=0 else 1/.85) for t in range(due)}
            row[shortage]=-1.; lp.inequality(row,-target)
        for d in range(4):
            debt=lp.var(cost=(w[2]+1.)/4); resource_indices.append(debt)
            lp.inequality({int(essence[-1][d]):-1.,debt:-1.},-1.)
        # Q is otherwise unpriced: an LP vertex can circulate tens of Mvar
        # between units at no business benefit. Among equal primary optima,
        # choose minimum absolute reactive effort (not a new risk objective).
        reactive_effort=[lp.absolute({int(j):1.}) for hour in qg for j in hour]
        result=lp.solve()
        if not result.success: return dict(success=False,mode=mode,message=result.message)
        primary_cost=np.asarray(lp.cost,float); primary_optimum=float(result.fun)
        lp.inequality(lp.dense_row(range(len(primary_cost)),primary_cost),primary_optimum+1e-7)
        lp.cost=[float(i in reactive_effort) for i in range(len(primary_cost))]
        refined=lp.solve()
        if refined.success: result=refined
        lp.cost=primary_cost.tolist()
        objective=float(primary_cost@result.x)
        x=result.x; volts=[(v0+dv@x[idx]).tolist() for idx,dv,v0 in voltage_rows]
        terms={name:float(sum(x[i]*lp.cost[i] for i in ix)) for name,ix in
            [('cost',planned_cost_indices),('service',service_indices),('resource',resource_indices),('change',change_indices)]}
        source_used={'wind':[], 'solar':[]}
        for t,exo in enumerate(forecast['values']):
            available=n.wind_share*exo[1]+n.solar_share*exo[2]
            fraction=np.divide(np.maximum(available-x[curtailments[t]],0.),available,
                out=np.zeros(66),where=available>1e-12)
            source_used['wind'].append((n.wind_share*exo[1]*fraction).tolist())
            source_used['solar'].append((n.solar_share*exo[2]*fraction).tolist())
        return dict(success=True,mode=mode,objective=objective,objective_terms=terms,
            power_mw=x[power[0]].tolist(),planned_power_mw=x[power].tolist(),planned_generation_mw=x[pg].tolist(),
            planned_q_mvar=x[qg].tolist(),planned_backup_mw=x[backups].tolist(),planned_ess_mwh=x[essence].tolist(),
            planned_base_fraction=(1.-x[sheds]).tolist(),planned_curtailment_mw=x[curtailments].tolist(),
            planned_wind_used_by_bus_mw=source_used['wind'],planned_solar_used_by_bus_mw=source_used['solar'],
            reactive_tiebreak=dict(success=bool(refined.success),primary_objective_increase=objective-primary_optimum,
                rule='lexicographic minimum sum absolute Q within 1e-7 of original business optimum'),
            planned_voltage_pu=volts,physics_version=VERSION,minimum_inequality_slack=float(np.min(result.ineqlin.residual)),
            approximation='radial affine voltage; conservative16-current polygon; native0.5MW cost secants; finite signed mode bank')

    def plan(self, view, forecast, weights, backup_remaining):
        self.calls+=1
        if not np.isclose(np.sum(weights),1.) or np.min(weights)<0: raise ValueError('Invalid preference weights')
        results=[self._mode(view,forecast,weights,backup_remaining,m) for m in ['charge','price','release']]
        good=[r for r in results if r['success']]
        if not good: raise RuntimeError('No finite MPC plan: '+str(results))
        best=min(good,key=lambda r:r['objective'])
        best['mode_attempts']=[dict(mode=r['mode'],success=r['success']) for r in results]
        return best
