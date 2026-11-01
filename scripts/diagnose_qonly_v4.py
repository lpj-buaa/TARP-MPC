"""Isolated feasibility witness for bounded Q with every non-slack P fixed."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandapower as pp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_voltage_v4 import minimal_result
from tarp37.collection import read, verify
from tarp37.data import ForecastTrace
from tarp37.network import UnifiedNetwork


def qonly(n, row, exo):
    point = row['routine_operating_point']
    net, gens, re, back, base, q, dev = n._build(
        exo, row['actual_device_mw'], point['base_fraction'], point['backup_mw'], False, point)
    for bus in net.bus.index:
        root = n.bus_ids[bus] in n.roots
        net.bus.at[bus, 'min_vm_pu'] = 1. if root else n.c.v_min
        net.bus.at[bus, 'max_vm_pu'] = 1. if root else n.c.v_max
    net.line['max_loading_percent'] = 100.
    for g, kind, idx in gens:
        if kind == 'sgen':
            pg = float(net.sgen.at[idx, 'p_mw'])
            net.sgen.at[idx, 'controllable'] = True
            net.sgen.at[idx, 'min_p_mw'] = pg
            net.sgen.at[idx, 'max_p_mw'] = pg
        # This auxiliary objective selects a Q-feasible witness, not a new MPC.
        pp.create_poly_cost(net, idx, kind, cp1_eur_per_mw=0., cp2_eur_per_mw2=1., cq2_eur_per_mvar2=.01)
    gen_indices = {idx for g, kind, idx in gens if kind == 'sgen'}
    for idx in net.sgen.index:
        if idx in gen_indices: continue
        pg = float(net.sgen.at[idx, 'p_mw'])
        net.sgen.at[idx, 'controllable'] = True
        net.sgen.at[idx, 'min_p_mw'] = pg
        net.sgen.at[idx, 'max_p_mw'] = pg
    try:
        pp.runopp(net, verbose=False, init='flat', numba=False, delta=1e-7,
                  OPF_VIOLATION=1e-7, OPF_FLOW_LIM=2, PDIPM_MAX_IT=200)
        # Remove only numerical zero-width bound excursions before independent PF.
        for idx in net.sgen.index:
            net.sgen.at[idx, 'p_mw'] = float(net.sgen.at[idx, 'min_p_mw'])
            net.sgen.at[idx, 'q_mvar'] = float(np.clip(net.res_sgen.at[idx, 'q_mvar'],
                net.sgen.at[idx, 'min_q_mvar'], net.sgen.at[idx, 'max_q_mvar']))
        pp.runpp(net, init='results', numba=False, max_iteration=100,
                 tolerance_mva=1e-10, calculate_voltage_angles=True)
        result = n._read(net, gens, re, back, base, q, dev)
        result['known'] = result['max_bus_residual_mva'] <= 1e-5
        for j, (g, kind, idx) in enumerate(gens):
            if kind == 'sgen':
                assert abs(result['generators'][j]['p_mw'] - point['generation_mw'][j]) < 1e-10
        result.update(status='independently_verified' if result['physical_valid'] else 'independent_check_failed',
                      voltage_min_pu=min(result['voltage']), voltage_max_pu=max(result['voltage']))
        return result
    except (pp.OPFNotConverged, pp.LoadflowNotConverged) as err:
        return dict(known=False, physical_valid=False, status='numerical_unknown', error=type(err).__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    run = project / 'artifacts/runs/risk37_formal_v4_s42_20261101'
    out = project / 'artifacts/diagnostics/voltage_20261101'
    original = read(out / 'probes.json')
    n = UnifiedNetwork(); trace = ForecastTrace(); cache = {}; results = []
    selected = original['cases'][:args.limit] if args.limit else original['cases']
    for i, case in enumerate(selected):
        key = case['key']; source = key['source_id']
        if source not in cache:
            marker = run / 'sources' / source / 'complete.json'
            verify(read(marker)); cache = {source: read(marker.parent / 'records.json')}
        rec = next(r for r in cache[source] if all(r[k] == key[k] for k in ['pool_hour', 'profile_id', 'candidate_id']))
        row = rec['rows'][key['future_index']]
        result = qonly(n, row, trace.actual(row['now']))
        results.append(dict(key=key, original=case['original'], q_only=minimal_result(result),
                            status=result['status'], voltage_min_pu=result.get('voltage_min_pu'),
                            voltage_max_pu=result.get('voltage_max_pu')))
        if args.limit or (i + 1) % 20 == 0:
            print(f'Q-only {i+1}/{len(selected)}: {result["status"]}', flush=True)
    positive = [r for r in results if r['original']['voltage_excess_pu'] > n.c.voltage_tolerance_pu]
    feasible = [r for r in positive if r['q_only']['physical_valid']]
    summary = dict(cases=len(results), sources=len(set(r['key']['source_id'] for r in results)),
                   pre_voltage_positive=len(positive), positive_full_feasible_with_only_Q=len(feasible),
                   positive_not_verified=len(positive)-len(feasible),
                   all_full_feasible=sum(bool(r['q_only']['physical_valid']) for r in results),
                   numerical_unknown=sum(r['status']=='numerical_unknown' for r in results),
                   unknown_is_not_infeasibility_proof=True,
                   all_native_P_Q_line_voltage_bounds_unchanged=True,
                   constant_local_P_RE_backup_base_load_and_EV_delivery=True)
    name = 'qonly_smoke.json' if args.limit else 'qonly.json'
    (out / name).write_text(json.dumps(dict(summary=summary, cases=results), indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
