"""Strong Sobol control in the actual family-dose candidate parameterization."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import qmc


def sequence(public,budget,seed):
    keys=sorted({p['key'] for p in public});pools={k:[p for p in public if p['key']==k] for k in keys};visits=defaultdict(int)
    count=int(np.ceil(budget/len(keys)))
    points={k:qmc.Sobol(2,scramble=True,seed=seed+1009*j).random_base2(int(np.ceil(np.log2(max(count,1))))) for j,k in enumerate(keys)}
    selected=[]
    for i in range(budget):
        key=keys[i%len(keys)];u=points[key][visits[key]];visits[key]+=1
        family='throttle' if u[0]<.5 else 'release'
        candidates=[p for p in pools[key] if p['family']==family] or pools[key]
        choice=min(candidates,key=lambda p:(abs(p['dose']-u[1]),p['id']))
        pools[key].remove(choice);selected.append(choice)
    return selected


def main():
    from tarp_boundary.io import PROJECT,read,write,sha,bind
    from tarp_boundary.metrics import evaluate
    from tarp_info.cache import Oracle
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();root=PROJECT/'artifacts/boundary_v2'/a.run_id
    bundle=read(root/'bundle.json');protocol=dict(code=sha(__file__),bundle=sha(root/'bundle.json'),budget=128,seeds=[42,43,44],
        coordinates='independent scrambled 2D Sobol in family x dose for each context; nearest unused candidate',training_calls=0)
    bind(root/'parameter_sobol_protocol.json',protocol);outputs={}
    for seed in [42,43,44]:
        path=root/'baselines'/f'parameter_sobol_s{seed}.json'
        if path.exists():result=read(path);assert result['protocol']==protocol
        else:
            oracle=Oracle(bundle['answers']);queries=[]
            for item in sequence(bundle['public'],128,seed):queries.append(dict(item,**oracle.query(item['id'])))
            result=dict(method='parameter_sobol',seed=seed,budget=128,queries=queries,oracle_calls=oracle.calls,
                protocol=protocol,acquisition_training_calls=0,risk_forecaster_training_calls=0,old_models_training_calls=0)
            result['evaluation']=evaluate(bundle,result);write(path,result)
        outputs[str(path)]=sha(path)
        e=next(x for x in result['evaluation'] if x['budget']==64);print('Parameter Sobol',seed,'64 budget',e['boundary_pairs_found'],'/',e['boundary_pairs_total'],'distance',e['distance'])
    write(root/'parameter_sobol_complete.json',dict(complete=True,outputs=outputs))


if __name__=='__main__':main()
