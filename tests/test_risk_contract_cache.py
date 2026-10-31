import json
import numpy as np
import pytest


def test_cached_old_physical_labels_cannot_bypass_fresh_packing_checks(tmp_path):
    from tarp37.packing import load_packed
    np.savez(tmp_path/'packed.npz',event=np.ones((1,4)))
    for name,value in [('schema',dict(physics_version='native37_pq_ac_0p9ka_economics_v3',
            risk_contract='four_physical_risks_0p93_1p07_v3')),('metadata',[]),('scalers',{})]:
        (tmp_path/(name+'.json')).write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError,match='physical/risk contract'):
        load_packed(tmp_path)


def test_old_checkpoint_is_rejected_before_constructing_or_running_a_model(tmp_path):
    import torch
    from tarp37.learning import predict_checkpoint
    from tarp37.contract import VERSION,RISK_VERSION
    torch.save(dict(schema=dict(physics_version='v3',risk_contract='v3')),tmp_path/'old.pt')
    packed=dict(schema=dict(physics_version=VERSION,risk_contract=RISK_VERSION))
    with pytest.raises(ValueError,match='physical/risk contract'):
        predict_checkpoint(tmp_path/'old.pt',packed,'cpu')


def test_native_query_rejects_an_old_reference_before_restoring_its_pool(tmp_path):
    from tarp37.query import run_query
    proposal=dict(reference=dict(role='train',context=dict(physics_version='v3',risk_contract='v3')))
    path=tmp_path/'proposal.json'; path.write_text(json.dumps(proposal),encoding='utf-8')
    with pytest.raises(ValueError,match='physical/risk contract'):
        run_query('unused',path,tmp_path/'output.json')


def test_unknown_diagnostic_masks_electrical_supervision_but_keeps_observed_service():
    from tarp37.packing import raw_targets
    from tarp37.contract import RISK_VERSION
    record=dict(rows=[dict(risk_contract=RISK_VERSION,pre_known=False,
        pre_voltage_excess_pu=.05,pre_capacity_ratio_peak=2.,queue_vehicle_min=12.)],
        event=[0.,0.,0.,0.],event_known=[False,False,True,False])
    target=raw_targets(record)
    assert not target['outcome_mask'][0,0] and not target['outcome_mask'][0,1]
    assert target['outcome_mask'][0,8] and target['outcome'][0,8]==12.
