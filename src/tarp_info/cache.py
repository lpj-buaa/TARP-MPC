"""A finite simulator cache with an explicit one-query access boundary."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from .profiles import describe,visible_state,context_key


class Oracle:
    def __init__(self,answers):self.__answers=answers;self.calls=[]
    def query(self,candidate_id):
        if candidate_id in self.calls:raise ValueError('Candidate already queried')
        value=deepcopy(self.__answers[candidate_id]);self.calls.append(candidate_id);return value


def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(source_project, output, utility_file):
    source=Path(source_project); out=Path(output);out.mkdir(parents=True,exist_ok=True)
    initial=[]
    for p in sorted((source/'artifacts/runs/risk37_formal_v4_s42_20261101/sources').glob('*/records.json')):
        initial.extend(r for r in load_json(p) if r['role']=='train')
    schema=load_json(source/'artifacts/retraining/risk37_td3_v4_s42_20261101/pack_initial/schema.json')
    envelope=np.asarray(schema['station_power_envelope_mw'])
    action_scale=np.r_[envelope,np.ones(4)]; utilities=load_json(utility_file)
    references={context_key(r):r for r in initial if r['candidate_id']==0}
    raw=[]
    for arm in ['td3','sobol','random']:
        for p in sorted((source/f'artifacts/retraining/risk37_{arm}_v4_s42_20261101/explore/queries').glob('*.json')):
            raw.append((str(p),load_json(p)))
    states=np.stack([visible_state(r['context']) for r in references.values()]); mean=states.mean(0);scale=np.maximum(states.std(0),.01)
    def public(r,id):
        return dict(id=id,key=context_key(r),source_id=r['source_id'],
            state=((visible_state(r['context'])-mean)/scale).tolist(),
            action=(np.asarray(r['context']['candidate_mw'])/action_scale).tolist())
    def answer(r,error):
        profile=describe(r,envelope); row=r['rows'][0]
        service=lambda x:x.get('departure_energy_deficit_kwh',0.)>.001 or x.get('fcs_wait_max_minutes',0.)>30.+1e-6
        ref=references[context_key(r)]
        return dict(profile=profile.tolist() if profile is not None else None,error=float(error),
            event=r['event'],known=r['event_known'],late_service=bool(not service(row) and any(service(x) for x in r['rows'][1:])),
            induced_joint=bool(all(r['event_known'][2:]) and all(r['event'][2:]) and not all(ref['event'][2:])))
    def signature(r):return (context_key(r),tuple(np.round(r['context']['candidate_mw'],9)))
    seen=set(); init=[]
    for i,r in enumerate(initial):
        id=f'initial_{i:04}'; seen.add(signature(r));a=answer(r,utilities['initial'][str(i)])
        init.append(dict(**public(r,id),**a))
    pub=[]; answers={}; origins={};duplicates=0
    for origin,r in raw:
        if signature(r) in seen:duplicates+=1;continue
        seen.add(signature(r));id=f'candidate_{len(pub):04}'
        pub.append(public(r,id));answers[id]=answer(r,utilities['queries'][origin]);origins[id]=origin
    bundle=dict(public=pub,initial=init,answers=answers,origins=origins,
        manifest=dict(initial_rows=len(init),query_pool_rows=len(pub),duplicates_removed=duplicates,
            sources=len(references)//4,contexts=len(references),profile_group_lengths=[108,15,20],
            descriptor='ESS directional resource envelope + future MPC block requests + preferences; not full feasible volume',
            state_scaler=dict(mean=mean.tolist(),scale=scale.tolist()),risk_forecaster_training_calls=0,
            scope='Retrospective finite-pool V4 simulator query experiment; no V5 performance claim'))
    (out/'cache.json').write_text(json.dumps(bundle,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return bundle
