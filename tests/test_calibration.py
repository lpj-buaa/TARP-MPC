import numpy as np


def test_equal_budget_calibration_checks_or_false_alarms():
    from tarp37.learning import alarm_thresholds
    rng=np.random.default_rng(4); scores=rng.normal(size=(200,4)); y=np.zeros((200,4),bool)
    tau,info=alarm_thresholds(scores,y,np.ones_like(y),.05)
    assert np.mean(np.any(scores>=tau,axis=1))<=.05+1e-10
    assert info['or_budget_estimable']


def test_complete_compound_recall_requires_every_true_risk_label():
    from tarp37.learning import summarize
    y=np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0]],bool)
    pred=np.array([[1,0,0,0],[1,1,0,0],[0,0,0,0]],bool)
    meta=[dict(source_id='a')]*3
    result=summarize(pred,y,np.ones_like(y),meta)
    assert result['complete_compound_recall_macro']==.5
