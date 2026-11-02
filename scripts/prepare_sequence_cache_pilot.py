"""Explicit six-source development pilot; no new simulation/LP and no formal-bank role changes."""
from pathlib import Path
import argparse,numpy as np
from tarp_boundary.io import read,write,PROJECT,sha
from tarp_sequence.data import boundary_pairs
from tarp_acq_evidence.pipeline import seal

p=argparse.ArgumentParser();p.add_argument('--run-id',default='sequence_cache_pilot_s42_v1');a=p.parse_args()
root=PROJECT/'artifacts/sequence_v4'/a.run_id
if root.exists():raise ValueError('Preserve existing pilot; choose another RunId')
bpath=PROJECT/'artifacts/acquisition_evidence_v3/evidence_formal_s42_v1/bundle.json';b=read(bpath)
allrows=b['initial']+[dict(r,**b['answers'][r['id']]) for r in b['public']];cases=[]
for k,key in enumerate(sorted({r['key'] for r in allrows})):
    rows=[r for r in allrows if r['key']==key];initial=[i for i,r in enumerate(rows) if r['id'] in {x['id'] for x in b['initial']}]
    case=dict(key=key,source_id=key,role='train' if k<4 else 'tune' if k==4 else 'confirm',
              state=np.r_[np.array(rows[0]['state'])/7,rows[0]['spatial']].tolist(),action=[r['action'] for r in rows],
              response=[r['response'] for r in rows],response_mask=np.ones((len(rows),60),bool).tolist(),
              gaps=[r['gap_mwh'] for r in rows],recoverable=[r['recoverable'] for r in rows],
              recovery_mask=np.ones((len(rows),3),bool).tolist(),descriptor=[r['descriptor'] for r in rows],valid=[True]*len(rows),
              families=[r['family'] for r in rows],doses=[r['dose'] for r in rows],ids=[r['id'] for r in rows],initial=initial)
    case['boundary_pairs']=boundary_pairs(rows,case['valid'],np.array(case['recoverable']));cases.append(case)
write(root/'dataset.json',dict(cases=cases,scope='six historical sources development cache pilot; not a new confirm experiment',
      source_sha=sha(bpath),counts=dict(contexts=6,branches=162,known_branches=162),native_simulations=0,mpc_solves=0))
seal(root/'dataset_complete.json',[root/'dataset.json'])
print(str(root))
