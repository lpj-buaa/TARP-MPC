"""Read-only V4 inference adapter. Run with PYTHONPATH pointing to frozen V4 src."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from tarp37.model import ConsequenceStudent
    from tarp37.packing import raw_targets,transform_context
    from tarp37 import assets,packing,model as model_module
    source=Path(a.source).resolve()
    if assets.PROJECT.resolve()!=source:raise ValueError('V4 inference adapter must import the frozen V4 project')
    root=source/'artifacts/retraining/risk37_td3_v4_s42_20261101'
    checkpoint=root/'fit_initial/checkpoint.pt';torch.set_num_threads(1)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ConsequenceStudent(saved['schema'],saved['decoder'],saved['width']);model.load_state_dict(saved['state_dict']);model.eval()
    read=lambda f:json.loads(f.read_text(encoding='utf-8'))
    initial=[]
    for f in sorted((source/'artifacts/runs/risk37_formal_v4_s42_20261101/sources').glob('*/records.json')):initial.extend(r for r in read(f) if r['role']=='train')
    refs={(r['source_id'],r['pool_hour'],r['profile_id']):r for r in initial if r['candidate_id']==0}
    def pred(r):
        features=transform_context(r['context'],r['context']['candidate_mw'],saved['scalers'])
        with torch.no_grad():return model({k:torch.tensor(v) for k,v in features.items()})['outcome'].numpy()[0]
    rp={k:pred(r) for k,r in refs.items()};scale=np.array(saved['scalers']['outcome']['scale'])
    def error(r):
        k=(r['source_id'],r['pool_hour'],r['profile_id']);t=raw_targets(r);rt=raw_targets(refs[k]);mask=t['outcome_mask']&rt['outcome_mask']
        delta=(t['outcome']-rt['outcome'])/scale;pp=pred(r)-rp[k]
        return float(np.abs(pp[mask]-delta[mask]).mean()) if mask.any() else 0.
    result=dict(initial={str(i):error(r) for i,r in enumerate(initial)},queries={})
    for arm in ['td3','sobol','random']:
        for f in sorted((source/f'artifacts/retraining/risk37_{arm}_v4_s42_20261101/explore/queries').glob('*.json')):result['queries'][str(f)]=error(read(f))
    inputs=[checkpoint,root/'pack_initial/scalers.json',root/'pack_initial/schema.json',Path(packing.__file__),Path(model_module.__file__)]
    result['frozen_inputs']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs}
    result['training_calls']=0;Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
    print('Frozen V4 forward only:',len(initial),'initial,',len(result['queries']),'query answers')


if __name__=='__main__':main()
