"""Independent source-level checks and consequence summaries after frozen reuse."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tarp37.collection import read, verify, write
from tarp37.contract import PhysicalContract
from tarp37.regulation_reuse import digest


def main():
    project = Path(__file__).resolve().parents[1]
    root = project / 'artifacts/regulation_reuse/routine_q_reuse_v5_s42_20261101'
    audit = read(root / 'audit.json'); complete = read(root / 'complete.json')
    assert digest(audit) == complete['fingerprint']
    verify(dict(inputs=audit['inputs'], outputs=complete['outputs']))
    contract = PhysicalContract.load(); old_root = Path(audit['source_run'])
    hardware = read(root / 'hardware_comparison.json'); now = contract.manifest()
    assert hardware['unchanged'] and not hardware['differences']
    for field in hardware['compared_fields']:
        assert now[field] == hardware['old_contract'][field]
    all_branches = []; roles = {}; unknown = Counter(); point_checks = Counter()
    extra_cost = Counter(); positive_cost_rows = []; unknown_by_source = {}; extra_shedding = []
    sources_with_compound = set(); states_with_compound = set()
    for marker in sorted((root / 'sources').glob('*/complete.json')):
        seal = read(marker); assert seal['fingerprint'] == complete['fingerprint']
        verify(dict(inputs={}, outputs=seal['outputs']))
        original_marker = old_root / 'sources' / marker.parent.name / 'complete.json'
        verify(read(original_marker))
        original = read(original_marker.parent / 'records.json')
        original_by_key = {(r['pool_hour'], r['profile_id'], r['candidate_id']):r for r in original}
        branches = read(marker.parent / 'branches.json'); rows = read(marker.parent / 'rows.json')
        all_branches.extend(branches)
        row_groups = defaultdict(list)
        for row in rows:
            key = (row['pool_hour'], row['profile_id'], row['candidate_id'])
            row_groups[key].append(row)
            source = original_by_key[key]['rows'][row['future_index']]
            normal = row['normal_regulation']; point = normal['point']
            if not normal['known']:
                assert point is None
                unknown[normal['status']] += 1
            if normal['physical_valid']:
                assert normal['known'] and point['physical_valid']
                assert point['max_bus_residual_mva'] <= 1e-5
                assert min(point['voltage']) >= contract.v_min - contract.voltage_tolerance_pu
                assert max(point['voltage']) <= contract.v_max + contract.voltage_tolerance_pu
                assert max(point['loading_ratio']) <= 1 + contract.capacity_ratio_tolerance
                for g, command, achieved in zip(contract.case.generators, source['routine_operating_point']['generation_mw'], point['generators']):
                    if g['bus'] not in contract.case.root_capacity_mw:
                        assert abs(command - achieved['p_mw']) < 1e-10
                pre = source['pre_routine_AC']
                assert np.allclose(point['renewable_used_by_bus_mw'], pre['renewable_used_by_bus_mw'], atol=1e-10, rtol=0)
                assert np.allclose(point['backup_by_feeder_mw'], pre['backup_by_feeder_mw'], atol=1e-10, rtol=0)
                assert np.allclose(point['net_p'], pre['net_p'], atol=1e-10, rtol=0)
                point_checks['valid_normal_points'] += 1
            delta = row['extra_recourse']['operating_cost_change']
            if row['extra_recourse'].get('additional_non_ev_shed_mwh', 0.) > 1e-8:
                extra_shedding.append(dict(source=row['source_id'],
                    additional=row['extra_recourse']['additional_non_ev_shed_mwh'],
                    normal_shed=normal['point']['non_ev_shed_mwh'],
                    numerical=source['correction_evidence']['numerical_fallback_used']))
            if delta is not None:
                sign = 'positive' if delta > 1e-8 else 'negative' if delta < -1e-8 else 'zero'
                extra_cost[sign] += 1
                if sign == 'positive':
                    numerical = source['correction_evidence']['numerical_fallback_used']
                    extra_cost['positive_with_numerical_fallback'] += numerical
                    extra_cost['positive_with_non_EV_shedding'] += source['non_ev_shed_mwh'] > 1e-8
                    positive_cost_rows.append(dict(source_id=source.get('source_id', row['source_id']),
                        pool_hour=row['pool_hour'], candidate_id=row['candidate_id'], future_index=row['future_index'],
                        cost_change=delta, numerical_fallback=numerical, non_ev_shed_mwh=source['non_ev_shed_mwh']))
        source_unknown = Counter()
        for b in branches:
            key = (b['pool_hour'], b['profile_id'], b['candidate_id'])
            original_record = original_by_key[key]
            assert b['event_old'] == original_record['event']
            assert b['event_new'][1:] == original_record['event'][1:]
            assert b['known_new'][1:] == original_record['event_known'][1:]
            group = row_groups[key]
            positive = any(r['normal_regulation']['known'] and r['normal_regulation']['point']['voltage_excess_pu'] > contract.voltage_tolerance_pu for r in group)
            known = positive or (len(group) == 6 and all(r['normal_regulation']['known'] for r in group))
            assert b['event_new'][0] == float(positive) and b['known_new'][0] == known
            assert not b['training_eligible']
            if not known:
                reasons = sorted(set(r['normal_regulation']['status'] for r in group if not r['normal_regulation']['known']))
                source_unknown[' + '.join(reasons)] += 1
            if b['event_new'][2] and b['event_new'][3]:
                sources_with_compound.add(b['source_id']); states_with_compound.add((b['source_id'], b['pool_hour']))
        unknown_by_source[marker.parent.name] = dict(source_unknown)
        roles.setdefault(branches[0]['role'], []).extend(branches)
    role_support = {}
    for role, b in roles.items():
        y = np.asarray([r['event_new'] for r in b], bool); k = np.asarray([r['known_new'] for r in b], bool)
        role_support[role] = dict(sources=len(set(r['source_id'] for r in b)), branches=len(b),
            positive=(y & k).sum(0).tolist(), negative=(~y & k).sum(0).tolist(), unknown=(~k).sum(0).tolist(),
            normal=int((k.all(1) & ~y.any(1)).sum()), service_RE_compound=int((y[:,2] & y[:,3]).sum()))
    references = {(b['source_id'], b['pool_hour'], b['profile_id']):b for b in all_branches if b['candidate_id'] == 0}
    transitions = defaultdict(list)
    for b in all_branches:
        if b['candidate_id'] == 0: continue
        a = np.asarray(b['event_new'], bool)
        ref = references[(b['source_id'], b['pool_hour'], b['profile_id'])]
        r = np.asarray(ref['event_new'], bool)
        for name, flag in [('service_induced', a[2] and not r[2]), ('service_mitigated', r[2] and not a[2]),
                           ('RE_induced', a[3] and not r[3]), ('RE_mitigated', r[3] and not a[3]),
                           ('compound_induced', a[2] and a[3] and not (r[2] and r[3])),
                           ('service_improved_RE_worsened', r[2] and not a[2] and a[3] and not r[3]),
                           ('RE_improved_service_worsened', r[3] and not a[3] and a[2] and not r[2])]:
            if flag: transitions[name].append(b)
    transition_summary = {name:dict(branches=len(b), sources=len(set(r['source_id'] for r in b)),
                         states=len(set((r['source_id'],r['pool_hour']) for r in b)),
                         families=dict(Counter(r['family'] for r in b))) for name,b in transitions.items()}
    verify(dict(inputs=audit['inputs'], outputs=complete['outputs']))
    reason_groups = Counter()
    for value in unknown_by_source.values():
        reason_groups.update(value)
    result = dict(sources=len(list((root/'sources').glob('*/complete.json'))), branches=len(all_branches),
        verified_unchanged_hardware_groups=len(hardware['compared_fields']),
        independent_normal_point_checks=dict(point_checks), unknown_hours=dict(unknown), unknown_branch_reasons=unknown_by_source,
        role_support=role_support, service_RE_compound=dict(branches=sum(bool(b['event_new'][2] and b['event_new'][3]) for b in all_branches),
            sources=len(sources_with_compound), states=len(states_with_compound)), paired_action_transitions=transition_summary,
        extra_cost_signs=dict(extra_cost), positive_cost_rows=positive_cost_rows,
        unknown_window_reason_groups=dict(reason_groups),
        additional_shedding_with_valid_normal=dict(hours=len(extra_shedding),
            sources=len(set(r['source'] for r in extra_shedding)),
            additional_mwh=sum(r['additional'] for r in extra_shedding),
            normal_shed_max=max((r['normal_shed'] for r in extra_shedding),default=0.),
            all_numerical_fallback=all(r['numerical'] for r in extra_shedding)),
        new_collection_calls=0, rl_training_calls=0, expert_training_calls=0, all_source_and_runtime_hashes_verified=True)
    write(root/'analysis/final_statistics.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['unknown_branch_reasons','positive_cost_rows']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
