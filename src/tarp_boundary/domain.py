"""Unchanged MPC constraint support and one-hour reserve recovery tasks."""
from copy import deepcopy
import importlib.util
import numpy as np


def reference_target(context):
    return np.array(context['view']['ess_energy_mwh'],float)


def restore_gap(template,terms,initial,target,ceiling,solve=None):
    lp=deepcopy(template);primary=np.asarray(lp.cost)
    lp.inequality(lp.dense_row(range(len(primary)),primary),ceiling)
    slacks=[]
    for expression,start,goal in zip(terms,initial,target):
        slack=lp.var(0.,None);slacks.append(slack)
        lp.inequality({**{k:-v for k,v in expression.items()},slack:-1.},float(start-goal))
    lp.cost=[float(i in slacks) for i in range(len(lp.cost))]
    result=(solve or type(template).solve)(lp)
    status=dict(success=bool(result.success),status=int(result.status),message=str(result.message))
    if not result.success:return dict(known=False,gap_mwh=None,recoverable=None,solver=status)
    gap=max(0.,float(np.sum(result.x[slacks])))
    return dict(known=True,gap_mwh=gap,recoverable=bool(gap<=1e-6),solver=status)


def measure(network,context,target,tolerance=.05):
    import tarp37.mpc as original
    spec=importlib.util.spec_from_file_location('tarp37._boundary_reader_private',original.__file__)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base=module.LP;captures=[]
    class Capture(base):
        def solve(self):
            result=super().solve()
            if result.success:self.last_success=result
            if not getattr(self,'captured',False):
                self.captured=True;captures.append((self,deepcopy(self),result))
            return result
    module.LP=Capture;controller=module.BusinessMPC(network);options=[];attempts=[]
    for mode in ['charge','price','release']:
        captures.clear()
        plan=controller._mode(context['view'],dict(values=np.array(context['visible_forecast'])),context['weights'],context['backup_energy_mwh'],mode)
        instance,lp,result=captures[0]
        attempts.append(dict(task='primary',mode=mode,status=int(result.status),success=bool(result.success)))
        if not result.success:
            if result.status==2:continue
            return dict(known=False,reason='primary numerical unknown',attempts=attempts)
        assert np.allclose(instance.last_success.x[:51],plan['planned_power_mw'][0],atol=1e-7)
        assert np.allclose(instance.last_success.x[51:61],plan['planned_generation_mw'][0],atol=1e-7)
        # Validate the affine energy expressions against the controller itself.
        coefficients=[network.c.ess_efficiency if lp.bounds[47+j][0]>=0 else 1/network.c.ess_efficiency for j in range(4)]
        first=np.asarray(context['view']['ess_energy_mwh'])+np.array(coefficients)*instance.last_success.x[47:51]
        assert np.allclose(first,plan['planned_ess_mwh'][0],atol=1e-7)
        options.append((mode,lp,float(plan['objective']),coefficients))
    if not options:return dict(known=False,reason='all modes infeasible',attempts=attempts)
    best=min(x[2] for x in options);ceiling=best+tolerance*abs(best)+1e-7
    directions={'SCS':list(range(37)),'FCS':list(range(37,47)),'ESS':list(range(47,51)),
        'PCC':[51+j for j,g in enumerate(network.gen) if g['bus'] in network.roots]}
    bounds={};gaps=[]
    for mode,template,cost,coefficients in options:
        if cost>ceiling+1e-5:continue
        recovery=restore_gap(template,[{47+j:coefficients[j]} for j in range(4)],
            context['view']['ess_energy_mwh'],target,ceiling,solve=base.solve)
        attempts.append(dict(task='recovery',mode=mode,**recovery['solver']))
        if not recovery['known']:return dict(known=False,reason='recovery numerical unknown',attempts=attempts)
        gaps.append(recovery['gap_mwh'])
        for name,indices in directions.items():
            for sign in [1.,-1.]:
                lp=deepcopy(template);primary=np.asarray(lp.cost)
                lp.inequality(lp.dense_row(range(len(primary)),primary),ceiling)
                lp.cost=[sign if i in indices else 0. for i in range(len(primary))]
                result=base.solve(lp)
                attempts.append(dict(task=name,mode=mode,side='min' if sign>0 else 'max',success=bool(result.success),status=int(result.status)))
                if not result.success:return dict(known=False,reason='endpoint unverified',attempts=attempts)
                bounds.setdefault(name,[]).append(float(result.x[indices].sum()))
    if not gaps:return dict(known=False,reason='no admissible mode',attempts=attempts)
    gap=min(gaps)
    return dict(known=True,bounds_mw={k:[min(v),max(v)] for k,v in bounds.items()},
        width_mw={k:max(v)-min(v) for k,v in bounds.items()},gap_mwh=gap,recoverable=bool(gap<=1e-6),
        reference_reserve_mwh=list(target),business_optimum=best,business_ceiling=ceiling,attempts=attempts,
        scope='linearized six-hour MPC support; one-hour reference reserve recovery within own +5% business objective')
