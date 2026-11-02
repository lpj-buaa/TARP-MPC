"""Frozen V4 forward on V5-visible features, aligned by physical field names."""
import argparse
from pathlib import Path
import numpy as np
import torch
from tarp_boundary.io import read,write,sha,bind,PROJECT
from tarp37.model import ConsequenceStudent
from tarp37.packing import visible_features,raw_targets,OUTCOMES


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--source',required=True);args=parser.parse_args()
    root=Path(args.root);source=Path(args.source);items=read(root/'inventory.json')
    folder=source/'artifacts/retraining/risk37_td3_v4_s42_20261101';checkpoint=folder/'fit_initial/checkpoint.pt'
    old_model=source/'src/tarp37/model.py';new_model=PROJECT/'src/tarp37/model.py'
    assert old_model.read_text(encoding='utf-8')==new_model.read_text(encoding='utf-8')
    inputs={str(f.resolve()):sha(f) for f in [checkpoint,folder/'pack_initial/scalers.json',folder/'pack_initial/schema.json',old_model,new_model,Path(__file__),PROJECT/'src/tarp37/packing.py']}
    inputs.update({x['path']:sha(x['path']) for x in items})
    bind(root/'teacher_protocol.json',dict(inputs=inputs,method='same-name actual physical outcome delta error; voltage definitions not interchanged',training_calls=0))
    if (root/'teacher.json').exists():return
    torch.set_num_threads(1);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ConsequenceStudent(saved['schema'],saved['decoder'],saved['width']);model.load_state_dict(saved['state_dict']);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    fields=[n for n in saved['schema']['outcome_fields'] if n in OUTCOMES and 'voltage_excess' not in n]
    old_columns=[saved['schema']['outcome_fields'].index(n) for n in fields];new_columns=[OUTCOMES.index(n) for n in fields]
    scale=np.asarray(saved['scalers']['outcome']['scale'])[old_columns]
    records={x['id']:read(x['path']) for x in items};predictions={};targets={}
    for ident,record in records.items():
        feature=visible_features(record['context'],record['context']['candidate_mw'])
        assert all(list(feature[k].shape)==v for k,v in saved['schema']['input_shapes'].items())
        for k in ['nodes','history','forecast','context','action']:
            scaler=saved['scalers'][k];feature[k]=(feature[k]-scaler['mean'])/scaler['scale']
        with torch.no_grad():predictions[ident]=model({k:torch.as_tensor(v[None],dtype=torch.float32) for k,v in feature.items()})['outcome'].numpy()[0][:,old_columns]
        target=raw_targets(record);targets[ident]=(target['outcome'][:,new_columns],target['outcome_mask'][:,new_columns])
    references={x['key']:x['id'] for x in items if x['family']=='reference'};errors={}
    for item in items:
        ident=item['id'];ref=references[item['key']];actual,mask=targets[ident];baseline,rm=targets[ref];mask=mask&rm
        difference=(actual-baseline)/scale-(predictions[ident]-predictions[ref])
        errors[ident]=float(np.abs(difference[mask]).mean()) if mask.any() else 0.
    for file,h in inputs.items():assert sha(file)==h
    write(root/'teacher.json',dict(errors=errors,fields=fields,frozen_inputs=inputs,training_calls=0,scope='legacy frozen physical-outcome model; not V5 warning performance'))
    print('Frozen forward complete:',len(errors),'branches;',len(fields),'same-name physical fields; training 0')


if __name__=='__main__':main()
