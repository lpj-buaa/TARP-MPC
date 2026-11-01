import numpy as np


def test_forecaster_predicts_delivered_response_instead_of_forcing_requested_power():
    import torch
    from tarp37.model import ConsequenceStudent
    schema=dict(input_shapes={'nodes':[66,13],'history':[10,12],'context':[20]},outcome_fields=['x'],adjacency=np.eye(66).tolist())
    model=ConsequenceStudent(schema,'gru',16)
    for p in model.parameters(): p.data.zero_()
    batch=dict(nodes=torch.zeros(2,66,13),history=torch.zeros(2,10,12),context=torch.zeros(2,20),
        action=torch.zeros(2,51),forecast=torch.zeros(2,6,4),initial_resource=torch.zeros(2,6),
        initial_response=torch.zeros(2,51),initial_weight=torch.ones(2,4)/4,candidate_request=torch.ones(2,51))
    assert torch.allclose(model(batch)['response'][:,0],torch.zeros(2,51)), 'site allocation is not delivered EV power'


def test_empty_or_future_targets_cannot_enter_visible_features():
    from tarp37.packing import visible_features
    from tarp37.contract import VERSION,RISK_VERSION
    import pytest
    with pytest.raises(ValueError,match='history'):
        visible_features(dict(history=[],physics_version=VERSION,risk_contract=RISK_VERSION),np.zeros(51))
