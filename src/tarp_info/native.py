"""Small V5 continuations from frozen V4 states; no retrospective relabeling."""
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np


def retry_q(solve):
    errors=[];last=None
    for init in ['flat','pf']:
        try:
            result=solve(init);result['normal_q_initialization']=init
            if result.get('known') and result.get('physical_valid'):return result
            last=result
        except RuntimeError as err:errors.append(err)
    if last is not None:return last
    raise errors[-1]


def regulator(network):
    import pandapower as pp
    from tarp37.regulation import RoutineQRegulator
    class RetriedRoutineQ(RoutineQRegulator):
        def _solve(self,exo,power,point):
            def once(init):
                n=self.n
                net,gens,re,back,base,q,dev=n._build(exo,power,point['base_fraction'],point['backup_mw'],False,point)
                for bus in net.bus.index:
                    root=n.bus_ids[bus] in n.roots
                    net.bus.at[bus,'min_vm_pu']=1. if root else n.c.v_min
                    net.bus.at[bus,'max_vm_pu']=1. if root else n.c.v_max
                net.line['max_loading_percent']=100.
                indices=set()
                for g,kind,idx in gens:
                    if kind=='sgen':indices.add(idx)
                    pp.create_poly_cost(net,idx,kind,cp1_eur_per_mw=0.,cp2_eur_per_mw2=1.,cq2_eur_per_mvar2=.01)
                for idx in net.sgen.index:
                    pg=float(net.sgen.at[idx,'p_mw'])
                    net.sgen.at[idx,'controllable']=True
                    net.sgen.at[idx,'min_p_mw']=pg;net.sgen.at[idx,'max_p_mw']=pg
                try:
                    pp.runopp(net,verbose=False,init=init,numba=False,delta=1e-7,OPF_VIOLATION=1e-7,OPF_FLOW_LIM=2,PDIPM_MAX_IT=200)
                    for idx in net.sgen.index:
                        net.sgen.at[idx,'p_mw']=float(net.sgen.at[idx,'min_p_mw'])
                        net.sgen.at[idx,'q_mvar']=float(np.clip(net.res_sgen.at[idx,'q_mvar'],net.sgen.at[idx,'min_q_mvar'],net.sgen.at[idx,'max_q_mvar']))
                    pp.runpp(net,init='results',numba=False,max_iteration=100,tolerance_mva=1e-10,calculate_voltage_angles=True)
                except (pp.OPFNotConverged,pp.LoadflowNotConverged) as err:raise RuntimeError(type(err).__name__) from err
                result=n._read(net,gens,re,back,base,q,dev);result['known']=result['max_bus_residual_mva']<=1e-5
                for j,g in enumerate(n.gen):
                    if g['bus'] not in n.roots and abs(result['generators'][j]['p_mw']-point['generation_mw'][j])>1e-10:raise RuntimeError('Active generation changed')
                result['base_fraction']=point['base_fraction'];return result
            try:return retry_q(once)
            except RuntimeError as err:raise pp.OPFNotConverged(str(err)) from err
    return RetriedRoutineQ(network)


def native_job(job):
    from tarp37.environment import UnifiedEnvironment,ExecutionUnknown
    from tarp37.risks import window_labels
    from .profiles import describe
    source,record,out=job;out=Path(out)
    if (out/'record.json').exists():raise FileExistsError('Preserve existing native branch')
    env=UnifiedEnvironment(out/'simulation',record['date'],int(record['source_id'].rsplit('_s',1)[1]))
    try:
        env.restore(Path(source)/'artifacts/runs/risk37_formal_v4_s42_20261101/sources'/record['source_id']/f"pool_{record['pool_hour']}")
        env.regulator=regulator(env.network);env.anchor=np.array(record['context']['preference_anchor'])
        ctx=env.context();ctx['candidate_mw']=record['context']['candidate_mw']
        rows=[];views=[]
        for h in range(6):
            views.append(env.context());plan=env.plan()
            try:rows.append(env.step(ctx['candidate_mw'] if h==0 else plan['power_mw'],plan))
            except ExecutionUnknown as error:rows.append(error.row);break
            print(f"{record['source_id']} a{record['candidate_id']} V5 hour {h+1}/6",flush=True)
        event,known=window_labels(rows,contract=env.c)
        output={k:record[k] for k in ['source_id','role','date','pool_hour','profile_id','candidate_id','family']}
        output.update(context=ctx,rows=rows,visible_views=views,event=event.tolist(),event_known=known.tolist(),
            origin='V4 stored physical state; six-hour V5 normal-Q continuation with bounded solver retries',training_eligible=False,
            profile=(describe(dict(rows=rows),env.c.station_power_envelope_mw).tolist() if len(rows)==6 and all(r['physical_valid'] for r in rows) else None),
            hardware=env.c.manifest())
        (out/'record.json').write_text(json.dumps(output,allow_nan=False),encoding='utf-8')
        return str(out/'record.json')
    finally:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--run-id',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    project=Path(__file__).resolve().parents[2];root=project/'artifacts/native_information'/a.run_id
    if root.exists():raise ValueError('Native RunId exists; preserve it and use a new RunId')
    root.mkdir(parents=True)
    jobs=[]
    for source_id in ['2024-03-04_s101','2024-04-01_s102']:
        records=json.loads((Path(a.source)/'artifacts/runs/risk37_formal_v4_s42_20261101/sources'/source_id/'records.json').read_text(encoding='utf-8'))
        for r in records:
            if r['pool_hour']==40 and r['profile_id']==0 and r['candidate_id'] in [0,1,2,4]:jobs.append((a.source,r,str(root/f"{source_id}_a{r['candidate_id']}")))
    assert len(jobs)==8
    (root/'protocol.json').write_text(json.dumps(dict(branches=8,workers=a.workers,source=a.source,
        source_states='two train initial states; profile0; candidate0/1/2/4; no selection based on outcomes',
        old_rl_training_calls=0,risk_forecaster_training_calls=0,experiment='V5 continuation sanity, not acquisition superiority')),encoding='utf-8')
    result=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(native_job,j) for j in jobs]):result.append(f.result())
    rows=[json.loads(Path(f).read_text(encoding='utf-8')) for f in result]
    summary=dict(complete=True,branches=len(rows),hours=sum(len(r['rows']) for r in rows),
        verified_hours=sum(x['physical_valid'] for r in rows for x in r['rows']),
        voltage_positive=sum(r['event_known'][0] and r['event'][0] for r in rows),
        voltage_unknown=sum(not r['event_known'][0] for r in rows),
        joint=sum(all(r['event_known'][2:]) and all(r['event'][2:]) for r in rows),paths=result)
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(summary)


if __name__=='__main__':main()
