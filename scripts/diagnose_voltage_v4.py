"""Read-only V4 voltage diagnosis; isolated AC probes never alter run labels."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandapower as pp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tarp37.collection import read, verify
from tarp37.data import ForecastTrace
from tarp37.network import UnifiedNetwork


def quantiles(values):
    return dict(zip(['min', 'p25', 'median', 'p75', 'p95', 'max'],
                    np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist()))


def affine(n, exo, power, point):
    load = n.load_share * exo[0] * point['base_fraction']
    available = {'wind': n.wind_share * exo[1], 'solar': n.solar_share * exo[2]}
    re = sum(np.minimum(available[s], np.maximum(point['renewable_used_by_source_bus_mw'][s], 0.))
             for s in available)
    p = load + n.device_map @ power - n.gen_map @ point['generation_mw'] - re - n.backup_map @ point['backup_mw']
    q = load * n.q_ratio - n.gen_map @ point['q_mvar']
    return 1. - n.r_voltage @ p - n.x_voltage @ q


def pv_probe(n, row, exo):
    point = row['routine_operating_point']
    net, gens, re, back, base, q, dev = n._build(
        exo, row['actual_device_mw'], point['base_fraction'], point['backup_mw'], False, point)
    controlled = []
    for j, (g, kind, index) in enumerate(gens):
        if kind == 'ext_grid':
            controlled.append((g, kind, index))
            continue
        # Same native generator, P command and Q bounds; only control mode differs.
        net.sgen.at[index, 'in_service'] = False
        ident = pp.create_gen(net, n.ix[g['bus']], p_mw=point['generation_mw'][j],
                              vm_pu=1., min_p_mw=g['pmin_mw'], max_p_mw=g['pmax_mw'],
                              min_q_mvar=g['qmin_mvar'], max_q_mvar=g['qmax_mvar'], name=g['id'])
        controlled.append((g, 'gen', ident))
    pp.runpp(net, init='flat', enforce_q_lims=True, numba=False,
             max_iteration=100, tolerance_mva=1e-9, calculate_voltage_angles=True)
    result = n._read(net, controlled, re, back, base, q, dev)
    result['known'] = result['max_bus_residual_mva'] <= 1e-5
    return result


def minimal_result(result):
    return {k: result.get(k) for k in ['known', 'physical_valid', 'voltage_excess_pu',
            'minimum_voltage_margin_pu', 'max_line_loading_ratio', 'generator_bound_excess',
            'pcc_export_permission_excess_mw', 'max_bus_residual_mva', 'generators', 'loss_mwh']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', default='risk37_formal_v4_s42_20261101')
    p.add_argument('--probes', action='store_true')
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project / 'artifacts/runs' / args.run_id
    out = project / 'artifacts/diagnostics/voltage_20261101'
    out.mkdir(parents=True, exist_ok=True)
    n = UnifiedNetwork(); c = n.c; trace = ForecastTrace()
    counts = dict(hours=0, under=0, over=0, pre_event=0, post_event=0,
                  forecast_applied_event=0, pre_and_forecast_event=0,
                  continuation_hours=0, continuation_forecast_affine_event=0,
                  continuation_forecast_affine_near_boundary=0)
    vectors = {k: [] for k in ['pre_excess', 'pre_min', 'pre_max', 'post_min', 'post_max',
                               'forecast_affine_min', 'forecast_AC_min', 'affine_minus_AC_min',
                               'local_q_change_l1', 'local_p_change_l1', 'q_fraction_peak']}
    sources = []; probes = []; seals = {}; grouped = {}
    for marker in sorted((root / 'sources').glob('*/complete.json')):
        verify(read(marker))
        records = read(marker.parent / 'records.json')
        seals[str(marker)] = hashlib.sha256(marker.read_bytes()).hexdigest()
        per = dict(source_id=records[0]['source_id'], role=records[0]['role'], hours=0, voltage_hours=0)
        source_candidates = []
        for rec in records:
            for h, row in enumerate(rec['rows']):
                if not row.get('pre_known'): continue
                key = dict(source_id=rec['source_id'], role=rec['role'], pool_hour=rec['pool_hour'],
                           profile_id=rec['profile_id'], candidate_id=rec['candidate_id'],
                           family=rec['family'], future_index=h, now=row['now'])
                pre = row['pre_routine_AC']; guard = row['forecast_allocation_guard']
                pv = np.asarray(pre['voltage']); postv = np.asarray(row['voltage'])
                event = pre['voltage_excess_pu'] > c.voltage_tolerance_pu
                forecast = np.asarray(guard['forecast_values'])
                av = affine(n, forecast, np.asarray(row['bounded_request_mw']), row['routine_operating_point'])
                diagnostic = guard['diagnostic']
                forecast_bad = guard['applied_diagnostic'].get('voltage_excess_pu', 0.) > c.voltage_tolerance_pu
                counts['hours'] += 1; per['hours'] += 1
                counts['under'] += int(pv.min() < c.v_min - c.voltage_tolerance_pu)
                counts['over'] += int(pv.max() > c.v_max + c.voltage_tolerance_pu)
                counts['pre_event'] += int(event); per['voltage_hours'] += int(event)
                counts['post_event'] += int(row['voltage_excess_pu'] > c.voltage_tolerance_pu)
                counts['forecast_applied_event'] += int(forecast_bad)
                counts['pre_and_forecast_event'] += int(event and forecast_bad)
                if h > 0 or rec['family'] == 'mpc_reference':
                    counts['continuation_hours'] += 1
                    counts['continuation_forecast_affine_event'] += int(av.min() < c.v_min - c.voltage_tolerance_pu or av.max() > c.v_max + c.voltage_tolerance_pu)
                    counts['continuation_forecast_affine_near_boundary'] += int(min(av.min() - c.v_min, c.v_max - av.max()) < 1e-5)
                    vectors['forecast_affine_min'].append(float(av.min()))
                    if diagnostic.get('known'):
                        vectors['forecast_AC_min'].append(float(min(diagnostic['voltage'])))
                        vectors['affine_minus_AC_min'].append(float(av.min() - min(diagnostic['voltage'])))
                local = [j for j, g in enumerate(n.gen) if g['bus'] not in n.roots]
                dq = np.asarray(row['correction_evidence']['q_change_mvar'])
                dp = np.asarray(row['correction_evidence']['generation_change_mw'])
                vectors['local_q_change_l1'].append(float(np.abs(dq[local]).sum()))
                vectors['local_p_change_l1'].append(float(np.abs(dp[local]).sum()))
                vectors['q_fraction_peak'].append(max(abs(row['routine_operating_point']['q_mvar'][j]) / n.gen[j]['qmax_mvar'] for j in local))
                for name, value in [('pre_excess', pre['voltage_excess_pu']), ('pre_min', pv.min()),
                                    ('pre_max', pv.max()), ('post_min', postv.min()), ('post_max', postv.max())]:
                    vectors[name].append(float(value))
                if rec['profile_id'] == 0:
                    source_candidates.append((key, row))
                    if rec['family'] == 'mpc_reference' and h in [0, 2, 5]:
                        probes.append((key, row))
                group_key = (rec['source_id'], rec['pool_hour'], rec['profile_id'], h)
                grouped.setdefault(group_key, []).append((event, float(pre['voltage_excess_pu'])))
        if source_candidates:
            # Add extreme checks, without choosing by any predictor performance.
            probes.append(max(source_candidates, key=lambda item: item[1]['pre_voltage_excess_pu']))
        sources.append(per)
    unique = {tuple(key.values()): (key, row) for key, row in probes}
    report = dict(run_id=args.run_id, counts=counts, source_counts=sources,
                  distributions={k: quantiles(v) for k, v in vectors.items()},
                  matched_hour_groups=len(grouped),
                  matched_hour_groups_with_event_disagreement=sum(len(set(e for e, v in g)) > 1 for g in grouped.values()),
                  probe_count=len(unique), input_seals=seals,
                  note='descriptive physical audit; no predictor selection or risk-label changes')
    (out / 'support.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['input_seals', 'source_counts']}), flush=True)
    if args.probes:
        results = []
        for i, (key, row) in enumerate(unique.values()):
            exo = trace.actual(row['now']); point = deepcopy(row['routine_operating_point'])
            # First test the Q from the already saved feasible execution, with all
            # other planned setpoints and actual traffic delivery held unchanged.
            point['q_mvar'] = [g['q_mvar'] for g in row['generators']]
            transplanted = n.pressure(exo, row['actual_device_mw'], point)
            try:
                regulated = pv_probe(n, row, exo)
                failure = None
            except pp.LoadflowNotConverged as err:
                regulated = dict(known=False, physical_valid=False)
                failure = type(err).__name__
            results.append(dict(key=key, original=minimal_result(row['pre_routine_AC']),
                                saved_Q_only=minimal_result(transplanted),
                                bounded_PV_only=minimal_result(regulated), error=failure,
                                constant_local_P_RE_backup_served_load_and_EV_delivery=True))
            if (i + 1) % 20 == 0: print(f'voltage probes {i+1}/{len(unique)}', flush=True)
        summary = dict(cases=len(results), sources=len(set(r['key']['source_id'] for r in results)),
                       pre_voltage_events=sum(r['original']['voltage_excess_pu'] > c.voltage_tolerance_pu for r in results))
        for variant in ['saved_Q_only', 'bounded_PV_only']:
            successful = [r for r in results if r[variant].get('known')]
            summary[variant] = dict(known=len(successful),
                remaining_voltage_events=sum(r[variant]['voltage_excess_pu'] > c.voltage_tolerance_pu for r in successful),
                full_physical_valid=sum(bool(r[variant]['physical_valid']) for r in successful),
                export_permission_excess=sum(r[variant]['pcc_export_permission_excess_mw'] > c.power_tolerance_mw for r in successful),
                generator_capability_excess=sum(r[variant]['generator_bound_excess'] > c.power_tolerance_mw for r in successful))
        (out / 'probes.json').write_text(json.dumps(dict(summary=summary, cases=results), indent=2), encoding='utf-8')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
