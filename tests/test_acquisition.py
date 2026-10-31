import numpy as np


def test_one_intervention_agent_updates_on_observed_learning_utility():
    from tarp37.acquisition import AcquisitionTD3
    agent=AcquisitionTD3(12,seed=1)
    before=[p.detach().clone() for p in agent.q1.parameters()]
    for j in range(8): agent.add(np.full(12,j/8),np.zeros(51),float(j))
    assert np.isfinite(agent.update())
    assert any(not np.allclose(a.numpy(),b.detach().numpy()) for a,b in zip(before,agent.q1.parameters()))


def test_acquisition_only_maps_to_feasible_station_directions():
    from tarp37.acquisition import physical_action
    ctx=dict(view=dict(charge_cap_mw=[.1]*51,discharge_cap_mw=[.05]*51,
        arrival_prior_mw=np.zeros((6,47)).tolist(),v2g_available_mwh=[.02]*47))
    p=physical_action(np.full(51,-1.),ctx,[.14]*37+[2.]*10)
    assert np.all(p[37:47]>=0) and np.min(p[:37])>=-.02


def test_seed_action_reward_is_not_assigned_to_a_different_clipped_action():
    from tarp37.acquisition import normalized_action,physical_action
    ctx=dict(view=dict(charge_cap_mw=[.1]*51,discharge_cap_mw=[.05]*51,
        arrival_prior_mw=np.zeros((6,47)).tolist(),v2g_available_mwh=[.02]*47))
    hardware=[.5]*37+[2.]*10
    outside=np.zeros(51); outside[0]=.5
    assert normalized_action(outside,ctx,hardware) is None
    power=physical_action(np.full(51,.5),ctx,hardware)
    action=normalized_action(power,ctx,hardware)
    assert action is not None and np.allclose(physical_action(action,ctx,hardware),power)


def test_completed_acquisition_cannot_silently_change_budget(tmp_path,monkeypatch):
    from tarp37 import acquisition
    from tarp37.collection import write
    folder=tmp_path/'artifacts/retraining/test/explore'; folder.mkdir(parents=True)
    write(folder/'complete.json',dict(inputs={},outputs={},policy='td3',episodes=3,acquisition_config={'seed':42}))
    monkeypatch.setattr(acquisition,'PROJECT',tmp_path)
    import pytest
    with pytest.raises(ValueError,match='changed'):
        acquisition.acquire('data','test',{'seed':42},policy='td3',episodes=4,resume=True)


def test_completed_acquisition_cannot_silently_switch_data(tmp_path,monkeypatch):
    from tarp37 import acquisition
    from tarp37.collection import write
    folder=tmp_path/'artifacts/retraining/test/explore'; folder.mkdir(parents=True)
    write(folder/'complete.json',dict(inputs={},outputs={},data_run='old',policy='td3',episodes=3,acquisition_config={'seed':42}))
    monkeypatch.setattr(acquisition,'PROJECT',tmp_path)
    import pytest
    with pytest.raises(ValueError,match='changed'):
        acquisition.acquire('new','test',{'seed':42},policy='td3',episodes=3,resume=True)


def test_completed_learning_cannot_silently_switch_data(tmp_path,monkeypatch):
    from tarp37 import learning
    from tarp37.collection import write
    source=tmp_path/'artifacts/runs/new/analysis'; source.mkdir(parents=True); write(source/'summary.json',{})
    folder=tmp_path/'artifacts/retraining/test/pack_initial'; folder.mkdir(parents=True)
    write(folder/'complete.json',dict(inputs={},outputs={},data_run='old',config={'seed':42}))
    monkeypatch.setattr(learning,'PROJECT',tmp_path)
    import pytest
    with pytest.raises(ValueError,match='changed'):
        learning.learning_task('new','test','pack_initial',{'seed':42},resume=True)


def test_incomplete_query_retry_retains_old_native_attempt(tmp_path):
    from tarp37.query import next_attempt
    original=next_attempt(tmp_path,'query_00000'); original.mkdir()
    (original/'partial.txt').write_text('retain failed run')
    retry=next_attempt(tmp_path,'query_00000')
    assert retry!=original and not retry.exists()
    assert (original/'partial.txt').read_text()=='retain failed run'


def test_partial_training_protocol_cannot_change_seed_or_dataset(tmp_path):
    from tarp37.training_protocol import ensure_protocol
    import pytest
    ensure_protocol(tmp_path,dict(data_run='a',config={'seed':42}))
    ensure_protocol(tmp_path,dict(data_run='a',config={'seed':42}))
    with pytest.raises(ValueError,match='changed'):
        ensure_protocol(tmp_path,dict(data_run='a',config={'seed':43}))
