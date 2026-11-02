"""Read-only support projections of admissible MPC plans, not AC set volumes.

Clone the exact MPC module into a private namespace so its builder is reused
without modifying the actual controller or globally monkeypatching its LP.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import argparse
import numpy as np


def project_options(network,context,cost_tolerance=.05,projection_solver=None):
    import tarp37.mpc as original
    spec=importlib.util.spec_from_file_location('tarp37._information_probe_private',original.__file__)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    base=module.LP;captures=[]
    class CapturingLP(base):
        def solve(self):
            result=super().solve()
            if result.success:self.last_success=result
            if not getattr(self,'captured',False):
                self.captured=True
                captures.append((self,deepcopy(self),result))
            return result
    module.LP=CapturingLP
    controller=module.BusinessMPC(network);options=[];mode_attempts=[];endpoint_attempts=[]
    forecast=dict(values=np.asarray(context['visible_forecast']))
    for mode in ['charge','price','release']:
        captures.clear()
        plan=controller._mode(context['view'],forecast,context['weights'],context['backup_energy_mwh'],mode)
        instance,lp,result=captures[0]
        mode_attempts.append(dict(mode=mode,success=bool(result.success),status=int(result.status),message=str(result.message)))
        if not plan['success']:
            if result.status==2:continue
            return dict(known=False,reason='MPC mode numerical unknown',mode_attempts=mode_attempts)
        # Check the builder's layout against its actual returned solution.
        # Refined Q tie-break is allowed to select another primary optimum.
        assert len(result.x)>=71
        assert np.allclose(instance.last_success.x[:51],np.asarray(plan['planned_power_mw'])[0],atol=1e-7)
        assert np.allclose(instance.last_success.x[51:61],np.asarray(plan['planned_generation_mw'])[0],atol=1e-7)
        # Independent names/order checks from the exact source contract.
        assert len(plan['planned_power_mw'][0])==51 and len(plan['planned_generation_mw'][0])==10
        options.append((mode,lp,float(plan['objective'])))
    if not options:return dict(known=False,reason='no linear MPC mode feasible')
    best=min(x[2] for x in options);ceiling=best+cost_tolerance*abs(best)+1e-7
    direction_indices={'SCS':range(37),'FCS':range(37,47),'ESS':range(47,51),
        'PCC':[51+j for j,g in enumerate(network.gen) if g['bus'] in network.roots]}
    bounds={};modes={};solve=projection_solver or base.solve
    for name,indices in direction_indices.items():
        endpoints=[]
        for mode,template,cost in options:
            if cost>ceiling+1e-5:continue
            pair=[]
            for sign in [1.,-1.]:
                lp=deepcopy(template);primary=np.array(lp.cost)
                lp.inequality(lp.dense_row(range(len(primary)),primary),ceiling)
                objective=np.zeros(len(primary));objective[list(indices)]=sign;lp.cost=objective.tolist()
                result=solve(lp)
                endpoint_attempts.append(dict(mode=mode,direction=name,side='min' if sign>0 else 'max',success=bool(result.success),status=int(result.status),message=str(result.message)))
                if not result.success:
                    return dict(known=False,reason='MPC endpoint unverified',mode_attempts=mode_attempts,endpoint_attempts=endpoint_attempts)
                value=float(result.x[list(indices)].sum())
                endpoints.append(value);pair.append(value)
            modes[mode]=True
        if not endpoints:return dict(known=False,reason='projection unverified')
        bounds[name]=[min(endpoints),max(endpoints)]
    return dict(known=True,layout_verified=True,best_business_objective=best,cost_tolerance=cost_tolerance,
        objective_ceiling=ceiling,bounds_mw=bounds,width_mw={k:v[1]-v[0] for k,v in bounds.items()},modes=list(modes),
        mode_attempts=mode_attempts,endpoint_attempts=endpoint_attempts,
        scope='support extrema of first actions admitting a six-hour linearized MPC plan within stated business-cost budget; union of native mode bank; not full AC feasibility or box volume')


def main():
    from tarp37.network import UnifiedNetwork
    p=argparse.ArgumentParser();p.add_argument('--native-run',default='v5_information_sanity_v1');a=p.parse_args()
    project=Path(__file__).resolve().parents[1];root=project/'artifacts/native_information'/a.native_run
    n=UnifiedNetwork();results=[]
    for f in sorted(root.glob('*/record.json')):
        rec=json.loads(f.read_text(encoding='utf-8'))
        for h in [1,3,5]:
            out=project_options(n,rec['visible_views'][h],.05)
            results.append(dict(source_id=rec['source_id'],action=rec['candidate_id'],future_hour=h,**out))
            print(rec['source_id'],rec['candidate_id'],h,out.get('width_mw'),flush=True)
    (root/'opportunity_projections.json').write_text(json.dumps(dict(records=results,probes=len(results),new_rollouts=0,training_calls=0),indent=2),encoding='utf-8')


if __name__=='__main__':main()
