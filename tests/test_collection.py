import pytest


def test_source_roles_are_isolated_by_more_than_complete_simulation_window():
    from tarp37.collection import validate_sources
    with pytest.raises(ValueError,match='overlap'):
        validate_sources([dict(date='2024-04-01',role='train'),dict(date='2024-04-02',role='confirm')])


def test_run_id_and_resume_do_not_overwrite_existing_outputs(tmp_path):
    from tarp37.collection import initialize
    initialize(tmp_path/'run',{'a':1},{})
    with pytest.raises(ValueError,match='Resume'):
        initialize(tmp_path/'run',{'a':1},{})
    initialize(tmp_path/'run',{'a':1},{},resume=True)
    with pytest.raises(ValueError,match='changed'):
        initialize(tmp_path/'run',{'a':2},{},resume=True)


def test_actual_consumed_vehicle_file_is_frozen():
    from tarp37.collection import simulation_inputs
    from tarp37.assets import TRAFFIC_SOURCE
    vehicle=next((TRAFFIC_SOURCE/'case').glob('*.veh.xml.gz'))
    assert str(vehicle) in simulation_inputs()


def test_all_unknown_analysis_retains_counts_and_null_physical_extrema(tmp_path):
    from tarp37.collection import write,stamp,analyze
    p=tmp_path/'sources/s1'; p.mkdir(parents=True)
    record=dict(source_id='s1',candidate_id=0,pool_hour=26,profile_id=0,
                event=[1,0,0,0],rows=[dict(physical_valid=False)])
    write(p/'records.json',[record]); write(p/'warmup.json',[])
    stamp(p,{},first_hour_checks=0)
    summary=analyze(tmp_path)
    assert summary['physical_valid_hours']==0 and summary['unknown_hours']==1
    assert summary['max_voltage_pu'] is None
