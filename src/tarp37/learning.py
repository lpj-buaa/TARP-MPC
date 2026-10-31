"""Scratch retraining adapters for existing graph/Transformer/GRU experts."""
from copy import deepcopy
from pathlib import Path
import json
import pickle
import random
import shutil
import re
import numpy as np
import torch
from torch.nn import functional as F
from .model import ConsequenceStudent,loss_terms,masked_huber
from .packing import INPUTS,OUTCOMES,pack_records,load_packed
from .collection import read,write,verify,stamp
from .assets import PROJECT,WORKSPACE,sha
from .training_protocol import ensure_protocol

EXPERTS=['full','local','wide_graph','service','resource','boundary']


def resolve_device(value):
    if value=='auto': return 'cuda' if torch.cuda.is_available() else 'cpu'
    if value.startswith('cuda') and not torch.cuda.is_available(): raise ValueError('CUDA unavailable in training Python')
    return value


def expert_schema(schema,name):
    s=deepcopy(schema); a=np.array(s['adjacency'])
    if name=='local': a=np.eye(len(a))
    if name=='wide_graph': a=a@a+np.eye(len(a)); a/=a.sum(1,keepdims=True)
    s['adjacency']=a.tolist(); return s


def mask_inputs(batch,name):
    b={k:v.clone() if k in INPUTS else v for k,v in batch.items()}
    if name=='service':
        b['nodes'][:,:,:3]=0.; b['context'][:,14:34]=0.
    elif name=='resource':
        b['nodes'][:,:,6:9]=0.
    elif name=='state_only':
        b['action'].zero_(); b['candidate_request'].zero_(); b['nodes'][:,:,11]=0.
    return b


def batches(packed,role,groups_per_batch=8,seed=None):
    groups={}; refs=packed['arrays']['reference']
    for i,m in enumerate(packed['metadata']):
        if m['role']==role: groups.setdefault(int(refs[i]),[]).append(i)
    keys=list(groups)
    if seed is not None: np.random.default_rng(seed).shuffle(keys)
    for start in range(0,len(keys),groups_per_batch):
        yield np.array([i for k in keys[start:start+groups_per_batch] for i in groups[k]],int)


def tensor_batch(packed,idx,device):
    lookup={int(i):j for j,i in enumerate(idx)}
    arrays=packed['arrays']; result={k:torch.as_tensor(v[idx],device=device) for k,v in arrays.items() if k!='reference'}
    result['reference']=torch.tensor([lookup[int(arrays['reference'][i])] for i in idx],device=device)
    return result


def train_expert(packed,name,out,config,device='auto'):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); device=resolve_device(device)
    seed=int(config.get('seed',42)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(int(config.get('torch_threads',1)))
    schema=expert_schema(packed['schema'],name)
    model=ConsequenceStudent(schema,'gru',int(config.get('width',64))).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config.get('learning_rate',.001)),weight_decay=.0001)
    train=np.array([m['role']=='train' for m in packed['metadata']]); e=packed['arrays']['event'][train]; mask=packed['arrays']['event_mask'][train]
    pos=torch.tensor(np.clip(((1-e)*mask).sum(0)/np.maximum((e*mask).sum(0),1),1,10),device=device)
    task_weight=torch.tensor([2.,2.,.5,.5] if name=='boundary' else [.5,.5,2.,.5] if name=='service' else [.5,.5,.5,2.] if name=='resource' else [1.]*4,device=device)
    best=float('inf'); best_state=None; stale=0; history=[]
    for epoch in range(int(config.get('epochs',100))):
        model.train(); values=[]
        for idx in batches(packed,'train',int(config.get('groups_per_batch',8)),seed+epoch):
            batch=tensor_batch(packed,idx,device); pred=model(mask_inputs(batch,name)); terms=loss_terms(pred,batch,True,pos)
            logits_loss=F.binary_cross_entropy_with_logits(pred['event'],batch['event'].float(),pos_weight=pos,reduction='none')*task_weight
            em=batch['event_mask'].bool(); event_loss=logits_loss[em].mean() if em.any() else pred['event'].sum()*0
            loss=terms['total']-terms['event']+event_loss
            if name=='boundary': loss+=masked_huber(pred['outcome'][:,:,:2],batch['outcome'][:,:,:2],batch['outcome_mask'][:,:,:2])
            if name=='resource': loss+=terms['resource']
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
            values.append(float(loss.detach()))
        model.eval(); scores=[]
        with torch.no_grad():
            for idx in batches(packed,'tune'):
                batch=tensor_batch(packed,idx,device); pred=model(mask_inputs(batch,name))
                ma=batch['outcome_mask'].bool(); refs=batch['reference']; dm=ma&ma[refs]&(refs!=torch.arange(len(refs),device=device))[:,None,None]
                score=(pred['outcome']-batch['outcome']).abs()[ma].mean()
                if dm.any(): score+=.5*((pred['outcome']-pred['outcome'][refs])-(batch['outcome']-batch['outcome'][refs])).abs()[dm].mean()
                scores.append(float(score))
        if not scores or not values: raise ValueError('Independent train and tune source groups are required')
        tune=float(np.mean(scores)); history.append(dict(epoch=epoch+1,train_loss=float(np.mean(values)),tune_score=tune))
        if tune<best-1e-7:
            best=tune; stale=0; best_state=deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else: stale+=1
        if stale>=int(config.get('patience',12)): break
    torch.save(dict(state_dict=best_state,schema=schema,scalers=packed['scalers'],expert=name,width=int(config.get('width',64)),decoder='gru'),out/'checkpoint.pt')
    write(out/'history.json',history); write(out/'fit.json',dict(expert=name,best_tune=best,epochs=len(history),seed=seed,device=device))
    stamp(out,{str(Path(__file__)):sha(__file__),str(PROJECT/'src/tarp37/model.py'):sha(PROJECT/'src/tarp37/model.py')},config=config)
    print(f'fit {name}: {len(history)} epochs, best tune={best:.4f}',flush=True)


def predict_checkpoint(path,packed,device='auto'):
    saved=torch.load(path,map_location='cpu',weights_only=False); device=resolve_device(device)
    model=ConsequenceStudent(saved['schema'],saved['decoder'],saved['width']).to(device); model.load_state_dict(saved['state_dict']); model.eval()
    result={}
    with torch.no_grad():
        for start in range(0,len(packed['metadata']),128):
            idx=np.arange(start,min(start+128,len(packed['metadata'])))
            batch={k:torch.as_tensor(packed['arrays'][k][idx],device=device) for k in INPUTS}
            pred=model(mask_inputs(batch,saved['expert']))
            for k,v in pred.items(): result.setdefault(k,[]).append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in result.items()}


def alarm_thresholds(scores,y,known,budget=.05,metadata=None):
    scores=np.asarray(scores); y=np.asarray(y,bool); known=np.asarray(known,bool)
    def cut(values,rows):
        weights=np.ones(len(values))/len(values)
        if metadata is not None:
            groups=np.array([metadata[i]['source_id'] for i in np.flatnonzero(rows)])
            unique=np.unique(groups)
            weights=np.array([1./len(unique)/np.count_nonzero(groups==g) for g in groups])
        order=np.argsort(values); values=values[order]; weights=weights[order]
        pivot=min(int(np.searchsorted(np.cumsum(weights),1-budget,side='left')),len(values)-1)
        value=values[pivot]
        return float(value+1e-10*max(1.,abs(value)))
    tau=[]; estimates=[]
    for k in range(4):
        rows=known[:,k]&~y[:,k]; neg=scores[rows,k]; estimates.append(bool(len(neg)))
        tau.append(cut(neg,rows) if len(neg) else .5)
    tau=np.array(tau); normal=known.all(1)&~y.any(1)
    if normal.any(): tau+=max(0.,cut(np.max(scores[normal]-tau,axis=1),normal))
    return tau,dict(budget=budget,per_risk_estimable=estimates,or_budget_estimable=bool(normal.any()),
        cal_or_fpr=float(np.mean(np.any(scores[normal]>=tau,axis=1))) if normal.any() else None,
        budget_unit='source_macro' if metadata is not None else 'row_micro',
        no_negative_fallback='score0.5 with budget explicitly unestimated')


def summarize(pred,y,known,metadata):
    pred=np.asarray(pred,bool); y=np.asarray(y,bool); known=np.asarray(known,bool)
    compound=known.all(1)&(y.sum(1)>=2); normal=known.all(1)&~y.any(1)
    hit=np.all(~y|pred,axis=1); sources=sorted(set(m['source_id'] for m in metadata)); macro=[]; fprs=[]; per_source={}
    for source in sources:
        idx=np.array([m['source_id']==source for m in metadata]); co=idx&compound; no=idx&normal
        value=float(np.mean(hit[co])) if co.any() else None
        if value is not None: macro.append(value)
        if no.any(): fprs.append(float(np.mean(pred[no].any(1))))
        per_source[source]=dict(complete_recall=value,compound_rows=int(co.sum()),normal_rows=int(no.sum()))
    per_risk=[]
    for k in range(4):
        pos=known[:,k]&y[:,k]; neg=known[:,k]&~y[:,k]; announced=known[:,k]&pred[:,k]
        per_risk.append(dict(recall=float(np.mean(pred[pos,k])) if pos.any() else None,
            fpr=float(np.mean(pred[neg,k])) if neg.any() else None,
            precision=float(np.mean(y[announced,k])) if announced.any() else None))
    return dict(complete_compound_recall_macro=float(np.mean(macro)) if macro else None,
        complete_compound_recall_micro=float(np.mean(hit[compound])) if compound.any() else None,
        or_fpr_macro=float(np.mean(fprs)) if fprs else None,or_fpr_micro=float(np.mean(pred[normal].any(1))) if normal.any() else None,
        compound_rows=int(compound.sum()),normal_rows=int(normal.sum()),sources=len(sources),per_risk=per_risk,per_source=per_source)


def evidence_features(packed,predictions):
    logits=np.stack([predictions[e]['event'] for e in EXPERTS],1)
    outcome=np.stack([predictions[e]['outcome'].max(1) for e in EXPERTS],1)
    references=packed['arrays']['reference']; delta=outcome-outcome[references]
    return np.concatenate([logits.reshape(len(logits),-1),outcome.reshape(len(logits),-1),delta.reshape(len(logits),-1),
        packed['arrays']['context'],packed['arrays']['action'],logits.std(1),logits.max(1)-logits.min(1)],1)


def fit_verifier(packed,predictions,out):
    from sklearn.ensemble import HistGradientBoostingClassifier
    x=evidence_features(packed,predictions); a=packed['arrays']; models=[]
    train=np.array([m['role']=='meta_train' for m in packed['metadata']])
    if not train.any(): raise ValueError('Independent meta_train sources required; expert in-sample predictions are excluded')
    for k in range(4):
        idx=train&a['event_mask'][:,k]; y=a['event'][idx,k]
        if not len(y): models.append(0.); continue
        if len(np.unique(y))==1: models.append(float(y[0])); continue
        model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,max_depth=3,l2_regularization=2.,learning_rate=.05,
                                            early_stopping=False,random_state=42)
        model.fit(x[idx],y); models.append(model)
    with Path(out).open('wb') as handle: pickle.dump(models,handle)


def verifier_scores(packed,predictions,path):
    x=evidence_features(packed,predictions)
    with Path(path).open('rb') as f: models=pickle.load(f)
    return np.stack([np.full(len(x),m) if isinstance(m,float) else m.predict_proba(x)[:,1] for m in models],1)


def source_bootstrap_difference(first,second,seed=42,draws=2000):
    ids=[s for s in first['per_source'] if first['per_source'][s]['complete_recall'] is not None]
    if len(ids)<2: return dict(sources=len(ids),difference=None,ci95=None)
    d=np.array([first['per_source'][s]['complete_recall']-second['per_source'][s]['complete_recall'] for s in ids])
    rng=np.random.default_rng(seed); values=np.mean(rng.choice(d,(draws,len(d)),replace=True),1)
    return dict(sources=len(ids),difference=float(d.mean()),ci95=np.quantile(values,[.025,.975]).tolist())


def learning_task(data_run,run_id,task,config,device='auto',resume=False,queries=None):
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',v) for v in [data_run,run_id]): raise ValueError('Invalid RunId')
    root=PROJECT/'artifacts/retraining'/run_id; source=PROJECT/'artifacts/runs'/data_run
    if not (source/'analysis/summary.json').exists(): raise ValueError('Complete physical collection/analysis first')
    folder=root/task; marker=folder/'complete.json'
    if marker.exists() and read(marker).get('data_run')!=data_run:
        raise ValueError('Learning source data changed; use a new learning RunId')
    inputs={str(p):sha(p) for p in [Path(__file__),PROJECT/'src/tarp37/model.py',PROJECT/'src/tarp37/packing.py']}
    for p in (source/'sources').glob('*/complete.json'): verify(read(p)); inputs[str(p)]=sha(p)
    protocol_inputs=dict(inputs)
    for name in ['acquisition.py','query.py','training_protocol.py','retrain_cli.py']:
        path=PROJECT/'src/tarp37'/name; protocol_inputs[str(path)]=sha(path)
    ensure_protocol(root,dict(data_run=data_run,config=config,inputs=protocol_inputs))
    if queries is not None:
        for p in Path(queries).glob('*.json'): inputs[str(p)]=sha(p)
    for other in root.glob('*/complete.json'):
        if other.parent!=folder: inputs[str(other)]=sha(other)
    packed_folder=None; packed_identity=None
    if task not in ('pack_initial','pack_augmented'):
        packed_folder=root/('pack_augmented' if task!='fit_initial' and (root/'pack_augmented/complete.json').exists() else 'pack_initial')
        packed_marker=packed_folder/'complete.json'; verify(read(packed_marker))
        inputs[str(packed_marker)]=sha(packed_marker)
        packed_identity=dict(packed_folder=str(packed_folder),packed_seal=sha(packed_marker),config=config)
        if folder.exists(): ensure_protocol(folder,packed_identity)
        if task=='predict' and read(root/'fit_experts/protocol.json')!=packed_identity:
            raise ValueError('Expert-bank packed data changed; use a new RunId')
    if marker.exists():
        saved=read(marker)
        # Later stages may now exist: validate the actual recorded dependency
        # set, rather than treating newly created downstream outputs as inputs.
        verify(saved)
        if saved.get('config')!=config: raise ValueError('Learning configuration changed; use a new learning RunId')
        print(task+': reused verified outputs',flush=True); return
    if folder.exists() and not resume: raise ValueError('Learning stage exists; use Resume or a new RunId')
    folder.mkdir(parents=True,exist_ok=True)
    if packed_identity is not None: ensure_protocol(folder,packed_identity)
    if task in ('pack_initial','pack_augmented'):
        records=[]
        for p in sorted((source/'sources').glob('*/records.json')): records+=read(p)
        if task=='pack_augmented':
            if queries is None: raise ValueError('RL query folder required for augmentation')
            for p in sorted(Path(queries).glob('*.json')):
                record=read(p)
                if record['role']!='train': raise ValueError('RL must not acquire confirm/meta/cal labels')
                records.append(record)
        pack_records(records,folder)
    else:
        packed=load_packed(packed_folder)
        if task=='fit_initial': train_expert(packed,'full',folder,config,device)
        elif task=='fit_experts':
            for name in [*EXPERTS,'state_only']:
                if (folder/name/'complete.json').exists():
                    saved=read(folder/name/'complete.json'); verify(saved)
                    if not saved.get('reused_initial') and saved.get('config')!=config:
                        raise ValueError('Partial expert-bank configuration changed; use a new RunId')
                    continue
                if name=='full' and packed_folder.name=='pack_initial':
                    verify(read(root/'fit_initial/complete.json')); target=folder/name; target.mkdir(parents=True,exist_ok=True)
                    for file in ['checkpoint.pt','fit.json','history.json']: shutil.copyfile(root/'fit_initial'/file,target/file)
                    stamp(target,{str(root/'fit_initial/complete.json'):sha(root/'fit_initial/complete.json')},reused_initial=True)
                    continue
                train_expert(packed,name,folder/name,config,device)
        elif task=='predict':
            for name in [*EXPERTS,'state_only']:
                pred=predict_checkpoint(root/'fit_experts'/name/'checkpoint.pt',packed,device)
                np.savez_compressed(folder/(name+'.npz'),**pred)
            initial=load_packed(root/'pack_initial')
            np.savez_compressed(folder/'initial_full.npz',**predict_checkpoint(root/'fit_initial/checkpoint.pt',initial,device))
        elif task=='fit_fusion':
            predictions={e:dict(np.load(root/'predict'/(e+'.npz'))) for e in EXPERTS}
            fit_verifier(packed,predictions,folder/'verifier.pkl')
        elif task in ('calibrate','evaluate'):
            predictions={e:dict(np.load(root/'predict'/(e+'.npz'))) for e in [*EXPERTS,'state_only']}
            sigmoid=lambda x:1/(1+np.exp(-np.clip(x,-40,40)))
            scores={e:sigmoid(p['event']) for e,p in predictions.items()}
            initial=sigmoid(np.load(root/'predict/initial_full.npz')['event'])
            scores['initial_full']=np.zeros((len(packed['metadata']),4)); scores['initial_full'][:len(initial)]=initial
            scores['uniform']=np.mean([scores[e] for e in EXPERTS],0)
            scores['union']=np.max([scores[e] for e in EXPERTS],0)
            scores['evidence_fusion']=verifier_scores(packed,predictions,root/'fit_fusion/verifier.pkl')
            a=packed['arrays']; y=a['event'].astype(bool); known=a['event_mask']; budget=float(config.get('budget',.05))
            if task=='calibrate':
                idx=np.array([m['role']=='cal' for m in packed['metadata']])
                if not idx.any(): raise ValueError('Independent calibration sources required')
                thresholds={}
                for name,s in scores.items():
                    cm=[m for m,j in zip(packed['metadata'],idx) if j]
                    tau,info=alarm_thresholds(s[idx],y[idx],known[idx],budget,cm); thresholds[name]=dict(threshold=tau.tolist(),**info)
                mt=np.array([m['role']=='meta_tune' for m in packed['metadata']])
                if not mt.any(): raise ValueError('Independent meta_tune sources required for best-single selection')
                values={}
                for name in EXPERTS:
                    mm=[m for m,j in zip(packed['metadata'],mt) if j]
                    tau,_=alarm_thresholds(scores[name][mt],y[mt],known[mt],budget,mm)
                    metric=summarize(scores[name][mt]>=tau,y[mt],known[mt],[m for m,j in zip(packed['metadata'],mt) if j])
                    values[name]=metric['complete_compound_recall_macro']
                best=max(EXPERTS,key=lambda k: values[k] if values[k] is not None else -1.)
                write(folder/'thresholds.json',dict(methods=thresholds,best_single=best,meta_tune_scores=values,budget=budget))
            else:
                calibrated=read(root/'calibrate/thresholds.json'); idx=np.array([m['role']=='confirm' for m in packed['metadata']])
                if not idx.any(): raise ValueError('Independent confirm sources required')
                meta=[m for m,j in zip(packed['metadata'],idx) if j]; metrics={}; alarms={}
                for name,s in scores.items():
                    alarm=s>=np.array(calibrated['methods'][name]['threshold']); alarms[name]=alarm[idx]
                    metrics[name]=summarize(alarm[idx],y[idx],known[idx],meta)
                best=calibrated['best_single']; metrics['best_single']=metrics[best]
                union=np.any([alarms[e] for e in EXPERTS],axis=0)
                single=np.any([np.all(~y[idx]|alarms[e],axis=1) for e in EXPERTS],axis=0)
                comp=known[idx].all(1)&(y[idx].sum(1)>=2)
                upper=dict(union_evidence_complete_coverage=float(np.mean(np.all(~y[idx]|union,1)[comp])) if comp.any() else None,
                    single_expert_oracle_complete_coverage=float(np.mean(single[comp])) if comp.any() else None,
                    scope='evidence availability ceiling, not deployed performance at the common OR budget')
                write(folder/'metrics.json',metrics); write(folder/'coverage.json',upper)
                write(folder/'comparison_ci.json',dict(vs_best_single=source_bootstrap_difference(metrics['evidence_fusion'],metrics[best]),
                    vs_full=source_bootstrap_difference(metrics['evidence_fusion'],metrics['full']),
                    vs_union=source_bootstrap_difference(metrics['evidence_fusion'],metrics['union'])))
        else: raise ValueError('Unknown learning task: '+task)
    stamp(folder,inputs,config=config,data_run=data_run,new_model_training_task=task in ['fit_initial','fit_experts','fit_fusion'],
          old_rl_training_calls=0,old_expert_training_calls=0)
