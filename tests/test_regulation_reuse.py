import hashlib
import json


def test_overlay_never_changes_input_and_is_not_a_new_closed_loop_dataset(tmp_path):
    from tarp37.regulation_reuse import audit_record
    from test_regulated_risks import rows
    rec = dict(source_id='a', role='train', date='2024-01-01', pool_hour=32, profile_id=0,
               candidate_id=0, family='mpc_reference', rows=rows(), event=[1.,0.,0.,0.], event_known=[True]*4)
    for x in rec['rows']:
        x.update(now=0, actual_device_mw=[0.]*51, routine_operating_point={},
                 physical_valid=True, total_operating_cost=1.)
    before = json.dumps(rec, sort_keys=True)
    class Regulator:
        def review(self, exo, power, point):
            return dict(known=True, physical_valid=False, point=dict(voltage_excess_pu=0.), status='voltage_ok_other_boundary')
    result, details = audit_record(rec, Regulator(), lambda now:[1.,0.,0.,30.])
    assert json.dumps(rec, sort_keys=True) == before
    assert result['event_new'] == [0.,0.,0.,0.]
    assert result['known_new'] == [True]*4
    assert not result['training_eligible']
    assert details[0]['extra_recourse']['operating_cost_change'] is None


def test_reuse_unknown_rows_cannot_become_safe_windows():
    from tarp37.regulation_reuse import audit_record
    from test_regulated_risks import rows
    rec = dict(source_id='a', role='train', date='2024-01-01', pool_hour=32, profile_id=0,
               candidate_id=0, family='mpc_reference', rows=rows(), event=[1.,0.,0.,0.], event_known=[True]*4)
    for x in rec['rows']:
        x.update(now=0, actual_device_mw=[0.]*51, routine_operating_point={}, physical_valid=True)
    class Regulator:
        def review(self,*args):
            return dict(known=False, physical_valid=False, point=None, status='normal_Q_numerical_unknown')
    result, _ = audit_record(rec, Regulator(), lambda now:[1.,0.,0.,30.])
    assert result['event_new'][0] == 0. and not result['known_new'][0]


def test_resume_rechecks_original_sealed_records_before_cached_return(tmp_path):
    import pytest
    from tarp37.assets import sha
    from tarp37.collection import write
    from tarp37.regulation_reuse import source_job
    source = tmp_path / 'source'; source.mkdir()
    record = source / 'records.json'; record.write_text('original')
    write(source / 'complete.json', dict(inputs={}, outputs={str(record):sha(record)},
        physics_version='native37_complete_plan_ac_hour_start_v4', risk_contract='four_boundary_events_complete_plan_v4'))
    output = tmp_path / 'derived'; output.mkdir()
    write(output / 'complete.json', dict(fingerprint='same', outputs={}))
    record.write_text('changed after original seal')
    with pytest.raises(ValueError, match='Frozen input or output changed'):
        source_job(source, output, 'same')
