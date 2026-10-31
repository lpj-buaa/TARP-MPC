from pathlib import Path
import hashlib
import sys

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent
NATIVE = Path('D:/tarp_ms_single_flow_test/external_repos/v2sim')
CASE = NATIVE/'cases/ux_37nodes'
TRAFFIC_SOURCE = WORKSPACE/'tarp_v2sim_engineering_grid/artifacts/normal_ev_5000_s42_v1'
PARQUET = WORKSPACE/'tarp_x_critical/.worktrees/v2sim-critical-td3/src/tarp_x_critical/data/processed/real_calibrated_15m_v1/temporal_15m.parquet'
PYTHON = WORKSPACE/'tarp_risk_discovery/.venv/Scripts/python.exe'
TRAIN_PYTHON = WORKSPACE/'tarp_x_expert_evaluation/.venv/Scripts/python.exe'


def activate():
    sys.dont_write_bytecode = True
    for p in [NATIVE, WORKSPACE/'tarp_v2sim_native_grid/src', WORKSPACE/'tarp_v2sim_traffic_diagnosis']:
        if str(p) not in sys.path: sys.path.insert(0, str(p))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def frozen_inputs():
    paths = list(CASE.glob('*.*'))
    paths += [p for p in (NATIVE/'v2sim').rglob('*.py') if '__pycache__' not in p.parts]
    paths += list((WORKSPACE/'tarp_rl_mpc_risk/.worktrees/r3/src').rglob('*.py'))
    paths += [WORKSPACE/'tarp_risk_discovery/assets/td3/gat_1500ev_3day_5000ep_best.pth']
    bank = WORKSPACE/'tarp_x_expert_evaluation/artifacts/runs/multihead_v61_formal_s42/banks'
    for name in ('*.pt', '*.pth', '*.pkl', '*manifest.json'):
        paths += list(bank.rglob(name))
    return {str(p): sha(p) for p in sorted(set(paths)) if p.is_file() and p.suffix != '.pyc'}
