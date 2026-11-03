from pathlib import Path
from copy import deepcopy
import json
import numpy as np


def fake_context(tmp_path,key,source,role,complete=True):
    folder=tmp_path/'contexts'/key;folder.mkdir(parents=True)
    record=folder/'record.json';record.write_text('{}')
    measurement=folder/'measurement.json';measurement.write_text('{}')
    from tarp_boundary.io import sha
    marker=folder/'complete.json'
    marker.write_text(json.dumps(dict(record_sha=sha(record),measurement_sha=sha(measurement))))
    ctx=dict(key=key,prototype=dict(source_id=source,role=role),items=[dict(id=key,source_id=source,role=role,path=str(record))])
    if complete:
        (folder/'complete_context.json').write_text(json.dumps(dict(complete=True,items=1,outputs={str(marker):sha(marker)})))
    return ctx


def test_only_sealed_complete_contexts_are_selected(tmp_path):
    from tarp_interim.data import select_contexts
    a=fake_context(tmp_path,'full','s102','train');b=fake_context(tmp_path,'partial','s1','train',False)
    rows=select_contexts([a,b],lambda c:tmp_path/'contexts'/c['key']/'complete_context.json',{'s102'})
    assert len(rows)==1 and rows[0]['items'][0]['role']=='development_test'
    assert a['items'][0]['role']=='train'


def test_original_confirm_never_enters_development_snapshot(tmp_path):
    from tarp_interim.data import select_contexts
    a=fake_context(tmp_path,'full','s1','confirm')
    assert not select_contexts([a],lambda c:tmp_path/'contexts'/c['key']/'complete_context.json',set())


def test_masked_monitoring_and_pair_deltas():
    from tarp_interim.training import regression_metrics
    truth=np.zeros((2,60));pred=truth.copy();pred[1,0]=2;truth[1,0]=1
    mask=np.ones_like(truth,dtype=bool);mask[0,1]=False;pred[0,1]=1e9
    m=regression_metrics(pred,truth,mask,np.array([0,0]))
    assert np.isclose(m['mae'],1/119)
    assert np.isclose(m['delta_mae'],1/59)


def test_rank_summary_skips_constant_truth():
    from tarp_interim.analysis import rank_summary
    assert rank_summary([0,0,0],[1,2,3])['spearman'] is None
    assert np.isclose(rank_summary([1,2,3],[3,2,1])['spearman'],-1)


def test_true_marginal_evaluation_does_not_reveal_queries():
    from tarp_interim.analysis import true_marginals
    from tarp_sequence.environment import AcquisitionEnv
    c=dict(initial=[0],valid=np.ones(3,bool),descriptor=np.array([[0.],[1.],[2.]]),boundary_pairs=[])
    e=AcquisitionEnv(c,2);before=list(e.seen);true_marginals(e)
    assert e.seen==before and e.queries==[] and e.remaining_budget==2


def test_monitored_fit_selects_only_original_tuning_roles(tmp_path):
    from tarp_interim.training import fit_response
    import torch
    def c(role):
        return dict(role=role,source_id=role,key=role,state=np.zeros(823),action=np.zeros((2,51)),response=np.zeros((2,60)),
                    response_mask=np.ones((2,60),bool),gaps=np.zeros((2,3)),recovery_mask=np.ones((2,3),bool),families=['reference','release'])
    fit_response([c('train'),c('tune'),c('development_test')],tmp_path,epochs=2)
    f=json.loads((tmp_path/'fit.json').read_text());assert f['train_sources']==['train'] and f['tune_sources']==['tune']
    assert f['optimizer_steps']==2 and len(f['history'])==2 and f['monitoring_random_draws']==0
    checkpoint=torch.load(tmp_path/'model.pt',map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==min(f['history'],key=lambda r:r['validation']['selection_loss'])['epoch']


def test_pause_inspection_reads_utf8_paths_in_windows_powershell(tmp_path):
    import shutil,subprocess,os
    import pytest
    if os.name!='nt':pytest.skip('Windows process-control interface')
    scripts=tmp_path/'scripts';scripts.mkdir()
    shutil.copyfile('scripts/control_sequence_pause.ps1',scripts/'control_sequence_pause.ps1')
    folder=tmp_path/'artifacts/sequence_v4/unit';folder.mkdir(parents=True)
    state=dict(status='paused',processes=[dict(pid=999999,suspended=True,created_at='none',command='D:\\博士资料包\\科研\\科研\\AAAI_RL\\worker.exe')])
    (folder/'pause_state.json').write_text(json.dumps(state,ensure_ascii=False),'utf-8')
    r=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(scripts/'control_sequence_pause.ps1'),'-Action','Inspect','-RunId','unit'],capture_output=True)
    assert r.returncode==0,r.stderr.decode(errors='replace')


def run_resume_fixture(tmp_path,inflight=False,locked=False):
    import shutil,subprocess
    scripts=tmp_path/'scripts';scripts.mkdir();shutil.copyfile('scripts/control_sequence_pause.ps1',scripts/'control_sequence_pause.ps1')
    folder=tmp_path/'artifacts/sequence_v4/unit';folder.mkdir(parents=True)
    entry=dict(pid=999999,suspended=True,created_at='none',command='not-a-real-process')
    if inflight:entry['resume_phase']='in_flight'
    (folder/'pause_state.json').write_text(json.dumps(dict(status='resuming',processes=[entry])),'utf-8')
    if locked:(folder/'pause_state.json.resume.lock').write_text('owned')
    return subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(scripts/'control_sequence_pause.ps1'),'-Action','Resume','-RunId','unit'],capture_output=True)


def test_resume_refuses_ambiguous_native_call(tmp_path):
    import os,pytest
    if os.name!='nt':pytest.skip('Windows process-control interface')
    r=run_resume_fixture(tmp_path,inflight=True)
    assert r.returncode!=0 and r.stderr.lower().startswith(b'ambiguous native resume state')


def test_resume_is_serialized_before_process_access(tmp_path):
    import os,pytest
    if os.name!='nt':pytest.skip('Windows process-control interface')
    r=run_resume_fixture(tmp_path,locked=True)
    assert r.returncode!=0 and b'exclusive resume lock' in r.stderr.lower()


def test_resume_roundtrip_on_owned_temporary_process(tmp_path):
    import ctypes,os,shutil,subprocess,pytest
    if os.name!='nt':pytest.skip('Windows native process integration')
    scripts=tmp_path/'scripts';scripts.mkdir();shutil.copyfile('scripts/control_sequence_pause.ps1',scripts/'control_sequence_pause.ps1')
    folder=tmp_path/'artifacts/sequence_v4/unit';folder.mkdir(parents=True)
    child=subprocess.Popen(['powershell','-NoProfile','-Command','Start-Sleep -Seconds 60'],creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        command=f"$q=Get-CimInstance Win32_Process -Filter 'ProcessId={child.pid}'; [pscustomobject]@{{pid=$q.ProcessId;created_at=$q.CreationDate.ToUniversalTime().ToString('o');command=$q.CommandLine;suspended=$true}} | ConvertTo-Json -Compress"
        info=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',command]))
        suspend=ctypes.WinDLL('ntdll').NtSuspendProcess;suspend.argtypes=[ctypes.c_void_p];suspend.restype=ctypes.c_long
        assert suspend(ctypes.c_void_p(int(child._handle)))==0
        (folder/'pause_state.json').write_text(json.dumps(dict(status='paused',processes=[info])),'utf-8')
        args=['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(scripts/'control_sequence_pause.ps1'),'-Action','Resume','-RunId','unit']
        first=subprocess.run(args,capture_output=True);assert first.returncode==0,first.stderr.decode(errors='replace')
        state=json.loads((folder/'pause_state.json').read_text('utf-8-sig'))
        assert state['status']=='resumed' and not state['processes'][0]['suspended'] and state['processes'][0]['resume_phase']=='complete'
        second=subprocess.run(args,capture_output=True);assert second.returncode==0 and b'Already resumed' in second.stdout
    finally:
        child.kill();child.wait()
