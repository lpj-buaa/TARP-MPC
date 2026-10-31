"""One real counterfactual query, always on a training-source native snapshot."""
import argparse
from pathlib import Path
import numpy as np
from .assets import PROJECT
from .collection import read,write
from .environment import UnifiedEnvironment,ExecutionUnknown
from .risks import window_labels


def next_attempt(parent,stem):
    """Retain incomplete native attempts and start a fresh restored branch."""
    parent=Path(parent)
    for i in range(10000):
        path=parent/(stem+f'_attempt_{i:04d}')
        if not path.exists(): return path
    raise RuntimeError('Too many retained native query attempts')


def run_query(data_run,proposal_path,output_path):
    proposal=read(proposal_path); reference=proposal['reference']
    if reference['role']!='train': raise ValueError('Acquisition is restricted to training sources')
    source=PROJECT/'artifacts/runs'/data_run/'sources'/reference['source_id']
    snapshot=source/f"pool_{reference['pool_hour']}"
    if not snapshot.is_dir(): raise ValueError('Frozen native pool snapshot missing')
    output_path=Path(output_path)
    if output_path.exists(): raise FileExistsError('Retain the existing query result')
    native=next_attempt(output_path.parent/'native',output_path.stem)
    env=UnifiedEnvironment(native,reference['date'],int(reference['source_id'].rsplit('_s',1)[1]))
    try:
        env.restore(snapshot); env.anchor=np.array(reference['context']['preference_anchor'])
        ctx=env.context(); ctx['candidate_mw']=proposal['power_mw']
        # The source state, resources, observed history and public forecast must
        # match the reference before obtaining a new label.
        for key in ['hour','now','bus_voltage_pu','backup_energy_mwh','history']:
            if ctx[key]!=reference['context'][key]: raise ValueError('Snapshot no longer matches the published source context: '+key)
        rows=[]
        for h in range(6):
            plan=env.plan()
            try: rows.append(env.step(proposal['power_mw'] if h==0 else plan['power_mw'],plan))
            except ExecutionUnknown as error: rows.append(error.row); break
        event,known=window_labels(rows)
        record={k:reference[k] for k in ['source_id','role','date','pool_hour','profile_id']}
        record.update(candidate_id=proposal['candidate_id'],family=proposal.get('family','td3_acquisition'),context=ctx,rows=rows,
                      event=event.tolist(),event_known=known.tolist())
        write(output_path,record)
    finally: env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-run',required=True); p.add_argument('--proposal',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); run_query(a.data_run,a.proposal,a.output)
