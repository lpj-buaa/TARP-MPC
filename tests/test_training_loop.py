"""Mechanical schema/training test on synthetic tensors, not empirical results."""
from copy import deepcopy
import numpy as np


def example_context(candidate):
    from tarp37.contract import VERSION
    history=[]
    for h in range(2):
        history.append(dict(grid_import_mw=8.,local_generation_mw=2.,loss_mwh=.1,voltage=[1.]*66,
            capacity_ratio=.4,queue_vehicle_min=0.,departure_energy_deficit_kwh=0.,ess_energy_mwh=[1.]*4,
            backup_energy_mwh=[6.]*2,weights=[.25]*4,actual_device_mw=[0.]*51,
            traffic_quarters=[dict(time_s=h*3600+q*900,station_mean_mw=[0.]*47) for q in range(4)]))
    view=dict(charge_cap_mw=[.1]*51,discharge_cap_mw=[.05]*51,need_mwh=[.02]*47,due_hour=[2.]*47,
        v2g_available_mwh=[.01]*47,previous_power_mw=[0.]*51,arrival_prior_mw=np.zeros((6,47)).tolist(),
        ess_energy_mwh=[1.]*4,queue_count=0)
    return dict(now=1700000000,hour=26,physics_version=VERSION,history=history,view=view,
        forecast_issued_at=np.full((6,3),1699999000).tolist(),visible_forecast=np.tile([10.,1.,2.,30.],(6,1)).tolist(),
        bus_net_p_mw=[.1]*66,bus_q_mvar=[.05]*66,bus_voltage_pu=[1.]*66,
        preference_anchor=[.25]*4,weights=[.25]*4,backup_energy_mwh=[6.]*2,
        routine_generation_mw=[1.]*10,routine_q_mvar=[.1]*10,candidate_mw=[candidate*.02]*51)


def example_records():
    from tarp37.contract import RISK_VERSION
    from tarp37.risks import window_labels
    records=[]
    for j,role in enumerate(['train','tune','meta_train','meta_tune','cal','confirm']):
        for candidate in [0,1]:
            rows=[]
            for h in range(6):
                rows.append(dict(risk_contract=RISK_VERSION,pre_known=True,pre_voltage_excess_pu=.001*candidate,
                    pre_capacity_ratio_peak=.5,max_line_loading_ratio=.4,grid_import_mw=8.,local_generation_mw=2.,loss_mwh=.1+.01*candidate,
                    voltage=[.99]*66,queue_vehicle_min=2.*candidate,departure_energy_deficit_kwh=0.,fcs_wait_max_minutes=31.*candidate,
                    non_ev_shed_mwh=0.,backup_mwh=0.,wind_available_mwh=1.,solar_available_mwh=1.,wind_curtail_mwh=.2*candidate,
                    solar_curtail_mwh=0.,total_operating_cost=2000.+candidate,allocation_capacity_reduction_mw=0.,
                    ess_energy_mwh=[1.]*4,backup_energy_remaining_mwh=[6.]*2,actual_device_mw=[candidate*.01]*51,
                    weights=[.25]*4,service_known=True))
            events,known=window_labels(rows)
            records.append(dict(source_id=f'fixture_{role}',role=role,date=f'2024-01-{1+7*j:02}',pool_hour=26,profile_id=0,
                candidate_id=candidate,family='fixture',context=example_context(candidate),rows=rows,event=events.tolist(),event_known=known.tolist()))
    return records


def test_pack_train_predict_and_conditional_evidence_shapes(tmp_path):
    from tarp37.packing import pack_records,visible_features
    from tarp37.learning import train_expert,predict_checkpoint,fit_verifier,verifier_scores,EXPERTS
    records=example_records(); packed=pack_records(records,tmp_path/'packed')
    assert packed['arrays']['history'].shape==(12,10,12)
    assert packed['arrays']['outcome'].shape==(12,6,19)
    context=deepcopy(records[0]['context']); other=deepcopy(context); other['future_truth']=[999.]*100
    assert all(np.array_equal(v,visible_features(other,np.zeros(51))[k]) for k,v in visible_features(context,np.zeros(51)).items())
    train_expert(packed,'full',tmp_path/'model',dict(epochs=1,patience=1,width=16,groups_per_batch=1,seed=42),'cpu')
    pred=predict_checkpoint(tmp_path/'model/checkpoint.pt',packed,'cpu')
    assert pred['outcome'].shape==(12,6,19) and np.isfinite(pred['event']).all()
    all_predictions={e:pred for e in EXPERTS}
    fit_verifier(packed,all_predictions,tmp_path/'verifier.pkl')
    assert verifier_scores(packed,all_predictions,tmp_path/'verifier.pkl').shape==(12,4)


def test_all_learning_stages_with_six_roles_and_immutable_resume(tmp_path,monkeypatch):
    import shutil
    from tarp37 import learning
    from tarp37.assets import PROJECT
    from tarp37.collection import write,stamp,read
    files=['model.py','packing.py','acquisition.py','query.py','training_protocol.py','retrain_cli.py']
    (tmp_path/'src/tarp37').mkdir(parents=True)
    for name in files: shutil.copyfile(PROJECT/'src/tarp37'/name,tmp_path/'src/tarp37'/name)
    source=tmp_path/'artifacts/runs/fixture'
    write(source/'analysis/summary.json',dict(scope='synthetic interface test'))
    records=example_records()
    for role in ['train','tune','meta_train','meta_tune','cal','confirm']:
        folder=source/'sources'/role
        write(folder/'records.json',[r for r in records if r['role']==role]); stamp(folder,{})
    monkeypatch.setattr(learning,'PROJECT',tmp_path)
    config=dict(epochs=1,patience=1,width=16,groups_per_batch=1,seed=42,budget=.05)
    tasks=['pack_initial','fit_initial','fit_experts','predict','fit_fusion','calibrate','evaluate']
    for task in tasks: learning.learning_task('fixture','test',task,config,'cpu',True)
    root=tmp_path/'artifacts/retraining/test'
    result=read(root/'evaluate/metrics.json')
    assert {'full','state_only','union','best_single','evidence_fusion'} <= set(result)
    assert result['full']['sources']==1
    for task in tasks: learning.learning_task('fixture','test',task,config,'cpu',True)
    assert all(read(root/task/'complete.json')['old_expert_training_calls']==0 for task in tasks)
    # An augmented pack must not silently replace the scaler underneath an
    # already trained or partially trained bank in the same learning run.
    import pytest
    learning.learning_task('fixture','test','pack_augmented',config,'cpu',True,tmp_path/'empty_queries')
    with pytest.raises(ValueError,match='changed'):
        learning.learning_task('fixture','test','fit_experts',config,'cpu',True)
