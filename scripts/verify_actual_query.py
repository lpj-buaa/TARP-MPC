"""One frozen-source native branch reproduced through the acquisition interface."""
import argparse
from pathlib import Path
import numpy as np
from tarp37.assets import PROJECT
from tarp37.collection import read,write
from tarp37.query import run_query


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-run',required=True); args=parser.parse_args()
    records=[]
    for p in sorted((PROJECT/'artifacts/runs'/args.data_run/'sources').glob('*/records.json')): records+=read(p)
    ref=next(r for r in records if r['role']=='train' and r['candidate_id']==0 and r['profile_id']==0)
    target=next(r for r in records if r['source_id']==ref['source_id'] and r['pool_hour']==ref['pool_hour'] and r['profile_id']==0 and r['candidate_id']==1)
    out=PROJECT/'artifacts/interface_verification'/args.data_run
    output=out/'query.json'
    if output.exists(): raise FileExistsError('Retain existing interface verification')
    reference={k:ref[k] for k in ['source_id','role','date','pool_hour','profile_id','context']}
    write(out/'proposal.json',dict(reference=reference,power_mw=target['context']['candidate_mw'],candidate_id='verification_throttle',family='interface_verification'))
    run_query(args.data_run,out/'proposal.json',output)
    observed=read(output)
    if len(observed['rows'])!=len(target['rows']): raise AssertionError('Replayed horizon differs')
    fields=['actual_device_mw','ess_energy_mwh','weights','voltage','backup_energy_remaining_mwh']
    errors={field:float(np.max(np.abs(np.array([r[field] for r in observed['rows']])-np.array([r[field] for r in target['rows']])))) for field in fields}
    if max(errors.values())>1e-6 or observed['event']!=target['event']: raise AssertionError('Branch query does not reproduce its frozen source: '+str(errors))
    write(out/'verification.json',dict(physical_hours=len(observed['rows']),max_errors=errors,matching_events=observed['event'],old_rl_training_calls=0,old_expert_training_calls=0))
    print(read(out/'verification.json'))


if __name__=='__main__': main()
