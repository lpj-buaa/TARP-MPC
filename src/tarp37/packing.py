"""Decision-time whitelist and new physical targets for the existing TARP-X student."""
from pathlib import Path
import json
import numpy as np
from .assets import PROJECT,sha
from .contract import PhysicalContract,RISK_VERSION,VERSION

INPUTS=['nodes','history','forecast','context','action','initial_resource','initial_response','initial_weight','candidate_request']
OUTCOMES=['pre_voltage_excess_pu','pre_capacity_ratio_peak','max_line_loading_ratio','grid_import_mw',
    'local_generation_mw','loss_mwh','v_min','v_max','queue_vehicle_min','departure_energy_deficit_kwh',
    'fcs_wait_max_minutes','non_ev_shed_mwh','backup_mwh','wind_available_mwh','solar_available_mwh',
    'wind_curtail_mwh','solar_curtail_mwh','total_operating_cost','allocation_capacity_reduction_mw']


def validate_schema_contract(schema):
    if schema.get('physics_version')!=VERSION or schema.get('risk_contract')!=RISK_VERSION:
        raise ValueError('Different physical/risk contract; regenerate V4 trajectories, labels and checkpoint')


def visible_features(ctx,candidate):
    history=ctx.get('history',[])
    if len(history)!=2 or any(len(x.get('traffic_quarters',[]))!=4 for x in history):
        raise ValueError('Two real grid hours and eight native quarter-hour history records required')
    issued=np.asarray(ctx['forecast_issued_at'])
    if np.any(issued>ctx['now']) or np.any(issued<=0): raise ValueError('Future or unknown publication time')
    c=PhysicalContract.load(); mapping={b:i for i,b in enumerate(c.case.buses)}
    db=np.array([mapping[b] for b in c.device_bus]); v=ctx['view']; action=np.array(candidate,float)
    nodes=np.zeros((66,13)); nodes[:,:3]=np.stack([ctx['bus_net_p_mw'],ctx['bus_q_mvar'],ctx['bus_voltage_pu']],-1)
    need=np.asarray(v['need_mwh']); due=np.asarray(v['due_hour'])
    local=[v['charge_cap_mw'],v['discharge_cap_mw'],np.r_[c.station_power_envelope_mw,np.ones(4)],
        np.r_[need,np.zeros(4)],np.r_[need*(due<=1),np.zeros(4)],np.r_[need*(due<=6),np.zeros(4)],
        np.r_[v['v2g_available_mwh'],np.zeros(4)],v['previous_power_mw'],action,
        np.r_[np.asarray(v['arrival_prior_mw'])[0],np.zeros(4)]]
    for j,values in enumerate(local): np.add.at(nodes[:,3+j],db,values)
    hs=[]
    for r in history:
        hs.append([r['grid_import_mw'],r['local_generation_mw'],r['loss_mwh'],min(r['voltage']),max(r['voltage']),
            r['capacity_ratio'],r['queue_vehicle_min'],r['departure_energy_deficit_kwh'],sum(r['ess_energy_mwh']),
            sum(r['backup_energy_mwh']),r['weights'][0],r['weights'][1]])
    for r in history:
        for quarter in r['traffic_quarters']:
            p=np.array(quarter['station_mean_mw']); phase=(quarter['time_s']%86400)/86400*2*np.pi
            hs.append([p[:37].sum(),p[37:].sum(),np.maximum(p,0).sum(),np.maximum(-p,0).sum(),
                np.count_nonzero(p>0)/47,np.count_nonzero(p<0)/47,*np.asarray(r['actual_device_mw'])[47:],np.sin(phase),np.cos(phase)])
    resource=np.r_[v['ess_energy_mwh'],ctx['backup_energy_mwh']]/[2.,2.,2.,2.,6.,6.]
    phase=(ctx['hour']%24)/24*2*np.pi
    context=np.r_[ctx['preference_anchor'],ctx['weights'],resource,ctx['routine_generation_mw'],ctx['routine_q_mvar'],
                  sum(need),v['queue_count'],np.sin(phase),np.cos(phase)]
    result=dict(nodes=nodes,history=np.array(hs),forecast=np.array(ctx['visible_forecast']),context=context,
        action=action,initial_resource=resource,initial_response=np.asarray(v['previous_power_mw']),
        initial_weight=np.array(ctx['weights']),candidate_request=action)
    if not all(np.isfinite(x).all() for x in result.values()): raise ValueError('Nonfinite visible observation')
    return result


def raw_targets(record):
    result={}
    for name,d in [('outcome',len(OUTCOMES)),('resource',6),('response',51),('weight',4)]:
        result[name]=np.zeros((6,d),np.float32); result[name+'_mask']=np.zeros((6,d),bool)
    for h,r in enumerate(record['rows'][:6]):
        if r.get('risk_contract')!=RISK_VERSION: raise ValueError('Old or mixed physical risk labels')
        r=dict(r)
        if 'voltage' in r: r.update(v_min=min(r['voltage']),v_max=max(r['voltage']))
        for j,f in enumerate(OUTCOMES):
            if f in ('pre_voltage_excess_pu','pre_capacity_ratio_peak') and not r.get('pre_known',False):
                continue
            value=r.get(f)
            if value is not None and np.isfinite(value): result['outcome'][h,j]=value; result['outcome_mask'][h,j]=True
        for key,fields,scale in [('resource',['ess_energy_mwh','backup_energy_remaining_mwh'],[2.,2.,2.,2.,6.,6.]),
                                 ('response',['actual_device_mw'],1.),('weight',['weights'],1.)]:
            offset=0
            for f in fields:
                value=r.get(f)
                length=4 if f=='ess_energy_mwh' else 2 if f=='backup_energy_remaining_mwh' else 51 if f=='actual_device_mw' else 4
                if value is not None:
                    result[key][h,offset:offset+length]=value; result[key+'_mask'][h,offset:offset+length]=True
                offset+=length
            result[key][h]/=scale
    result['event']=np.array(record['event'],np.float32); result['event_mask']=np.array(record['event_known'],bool)
    return result


def _fit(values,mask=None):
    flat=values.reshape(-1,values.shape[-1]); valid=np.ones_like(flat,bool) if mask is None else mask.reshape(flat.shape)
    means=[]; scales=[]
    for j in range(flat.shape[-1]):
        x=flat[valid[:,j],j]
        means.append(float(np.mean(x)) if len(x) else 0.); scales.append(float(max(np.std(x),1e-3)) if len(x) else 1.)
    return dict(mean=means,scale=scales)


def pack_records(records,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if any(r['context'].get('physics_version')!=VERSION for r in records):
        raise ValueError('Different physical/operating contract; regenerate trajectories before retraining')
    features=[visible_features(r['context'],r['context']['candidate_mw']) for r in records]
    targets=[raw_targets(r) for r in records]
    arrays={k:np.stack([f[k] for f in features]).astype(np.float32) for k in INPUTS}
    arrays.update({k:np.stack([t[k] for t in targets]) for k in targets[0]})
    metadata=[{k:r[k] for k in ['source_id','role','date','pool_hour','profile_id','candidate_id','family']} for r in records]
    train=np.array([m['role']=='train' for m in metadata]); scalers={}
    if not train.any(): raise ValueError('Training sources required for scalers')
    for name in ['nodes','history','forecast','context','action','outcome']:
        scalers[name]=_fit(arrays[name][train],arrays['outcome_mask'][train] if name=='outcome' else None)
        s=scalers[name]; arrays[name]=(arrays[name]-np.array(s['mean']))/np.array(s['scale'])
        arrays[name]=arrays[name].astype(np.float32)
    lookup={(r['source_id'],r['pool_hour'],r['profile_id']):i for i,r in enumerate(records) if r['candidate_id']==0}
    arrays['reference']=np.array([lookup[(r['source_id'],r['pool_hour'],r['profile_id'])] for r in records],np.int64)
    c=PhysicalContract.load(); adj=np.eye(66)
    for l in c.case.lines:
        i,j=list(c.case.buses).index(l['from_bus']),list(c.case.buses).index(l['to_bus']); adj[i,j]=adj[j,i]=1
    adj/=adj.sum(1,keepdims=True)
    schema=dict(input_shapes={k:list(arrays[k].shape[1:]) for k in INPUTS},outcome_fields=OUTCOMES,adjacency=adj.tolist(),
        station_power_envelope_mw=c.station_power_envelope_mw.tolist(),
        physics_version=VERSION,risk_contract=RISK_VERSION,risk_names=['Voltage','NetworkCapacity','ChargingService','RenewableCurtailment'],
        model_provenance='Existing ConsequenceStudent graph/Transformer/GRU architecture; realized first-hour response learned instead of forced allocation')
    support={}
    for role in sorted({m['role'] for m in metadata}):
        idx=np.array([m['role']==role for m in metadata])
        support[role]=dict(rows=int(idx.sum()),sources=len({m['source_id'] for m in metadata if m['role']==role}),risks={})
        for j,name in enumerate(schema['risk_names']):
            known=arrays['event_mask'][idx,j]; positive=arrays['event'][idx,j]>.5
            support[role]['risks'][name]=dict(positive=int((known&positive).sum()),negative=int((known&~positive).sum()),unknown=int((~known).sum()))
    np.savez_compressed(output/'packed.npz',**arrays)
    for name,value in [('metadata',metadata),('schema',schema),('scalers',scalers),('label_support',support)]:
        (output/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return load_packed(output)


def load_packed(folder):
    folder=Path(folder)
    info={k:json.loads((folder/(k+'.json')).read_text(encoding='utf-8')) for k in ['metadata','schema','scalers']}
    validate_schema_contract(info['schema'])
    return dict(arrays={k:v for k,v in np.load(folder/'packed.npz').items()},**info)


def transform_context(ctx,candidate,scalers):
    features=visible_features(ctx,candidate)
    for k in ['nodes','history','forecast','context','action']:
        features[k]=(features[k]-scalers[k]['mean'])/scalers[k]['scale']
    return {k:np.asarray(v,np.float32)[None] for k,v in features.items()}
