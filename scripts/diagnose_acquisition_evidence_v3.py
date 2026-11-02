"""Frozen post-hoc diagnostics; no optimizer, simulator or new query."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from tarp_boundary.io import read,write
from tarp_acq_evidence.analysis import summary,paired
from tarp_acq_evidence.data import RESPONSE_GROUPS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root)
    records=defaultdict(list);errors=defaultdict(list);drift=[];remaining_positive={};bundle=read(root/'bundle.json')
    for path in (root/'experiments/matched').glob('*/result.json'):
        r=read(path);method=r['method'];snaps={s['budget']:s for s in r['rankings']};s=snaps[64]
        if method=='stable':remaining_positive[str(r['seed'])]=sum(any(g>1e-6 for g in bundle['answers'][ident]['gap_mwh']) for ident in s['candidate_ids'])
        for metric in ['spearman','top3_capture','regret']:
            for row in s['sources']:
                records[metric].append(dict(method=method+'_q',seed=r['seed'],source_id=row['source_id'],value=row[metric]))
            if method in ['stable','combined']:
                for row in s['derived_sources']:
                    records[metric].append(dict(method=method+'_derived',seed=r['seed'],source_id=row['source_id'],value=row[metric]))
        if method in ['stable','combined']:
            for name,(lo,hi) in RESPONSE_GROUPS.items():
                y=np.array([bundle['answers'][ident]['response'][lo:hi] for ident in s['candidate_ids']]);pred=np.array(s['response_predictions'])[:,lo:hi]
                for key in sorted(set(s['keys'])):
                    ix=np.array(s['keys'])==key
                    for model,err in [(method,np.abs(pred-y).mean(1)),('reference_zero',np.abs(y).mean(1))]:
                        # Reference-zero rows repeated once for each paired model; mean unaffected.
                        errors[name].append(dict(method=model,seed=r['seed'],source_id=key,value=float(err[ix].mean())))
        if method=='old_summary':
            prev=snaps[32];past=dict(zip(prev['candidate_ids'],prev['truth']))
            for key in sorted(set(s['keys'])):
                ids=[ident for ident,k in zip(s['candidate_ids'],s['keys']) if k==key]
                lookup=dict(zip(s['candidate_ids'],s['truth']));x=np.array([past[i] for i in ids]);y=np.array([lookup[i] for i in ids])
                rho=float(spearmanr(x,y).statistic) if np.ptp(x)>0 and np.ptp(y)>0 else None
                drift.append(dict(seed=r['seed'],source_id=key,source_rank_correlation_32_to64=rho,
                                  fraction_utility_halved=float(np.mean(y<x/2)),mean_utility_32=float(x.mean()),mean_utility_64=float(y.mean())))
    result=dict(score_readouts={m:summary(v) for m,v in records.items()},response_vs_reference={m:summary(v) for m,v in errors.items()},
                readout_comparisons={m:[paired(v,a,b) for a,b in [('stable_derived','stable_q'),('combined_derived','combined_q'),
                                                                 ('stable_derived','old_summary_q')]] for m,v in records.items()},
                target_drift=drift,remaining_gap_positive_by_seed=remaining_positive,extra_training_calls=0,new_physical_simulations=0,
                scope='secondary readout diagnostic on frozen matched64 predictions, not an active-policy victory')
    write(root/'diagnostics.json',result)
    for m,v in result['score_readouts'].items():print(m,{k:round(x['mean'],4) for k,x in v.items()})
    print('derived minus q:',result['readout_comparisons']['spearman'])


if __name__=='__main__':main()
