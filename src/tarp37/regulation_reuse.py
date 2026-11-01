"""Immutable V4-state reuse for V5 normal-Q label support, never training."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import os

import numpy as np

from .assets import PROJECT, sha
from .collection import read, verify, write
from .contract import VERSION, RISK_VERSION
from .regulation import RoutineQRegulator, extra_recourse
from .risks import window_labels


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.pending')
    write(temporary, value); temporary.replace(path)


def audit_record(record, regulator, actual):
    rows = []; details = []
    keys = ['source_id', 'role', 'date', 'pool_hour', 'profile_id', 'candidate_id', 'family']
    meta = {k: record[k] for k in keys}
    for h, source in enumerate(record['rows']):
        normal = regulator.review(actual(source['now']), source['actual_device_mw'], source['routine_operating_point'])
        overlay = dict(source, regulation_known=normal['known'],
                       regulated_voltage_excess_pu=normal['point']['voltage_excess_pu'] if normal['known'] else None)
        rows.append(overlay)
        details.append(dict(**meta, future_index=h, now=source['now'],
                            normal_regulation=normal, extra_recourse=extra_recourse(normal, source),
                            raw_voltage_excess_pu=source.get('pre_voltage_excess_pu'),
                            actual_recovery_cost=source.get('recovery_cost'),
                            actual_non_ev_shed_mwh=source.get('non_ev_shed_mwh'),
                            actual_backup_mwh=source.get('backup_mwh'),
                            planned_correction_evidence=source.get('correction_evidence'),
                            comparison_scope='V4 observed execution minus V5 normal-Q counterfactual; not V5 closed-loop outcomes'))
    events, known = window_labels(rows)
    if not np.array_equal(events[1:], record['event'][1:]) or not np.array_equal(known[1:], record['event_known'][1:]):
        raise RuntimeError('A non-Voltage risk label changed during immutable reuse')
    result = dict(**meta, event_old=record['event'], known_old=record['event_known'],
                  event_new=events.tolist(), known_new=known.tolist(),
                  label_contract=RISK_VERSION, source_physics_version='native37_complete_plan_ac_hour_start_v4',
                  execution_scope='V4 fixed trajectories with offline V5 normal-Q review', training_eligible=False)
    return result, details


class CachedRegulator:
    def __init__(self, network, folder, fingerprint):
        self.controller = RoutineQRegulator(network)
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint

    def review(self, exo, power, point):
        key = digest(dict(exo=np.asarray(exo).tolist(), power=power, point=point, fingerprint=self.fingerprint))
        path = self.folder / (key + '.json')
        if path.exists():
            data = read(path)
            if data['checksum'] != digest(data['result']):
                raise ValueError('Unsealed Q-cache checksum mismatch')
            return data['result']
        result = self.controller.review(exo, power, point)
        atomic(path, dict(result=result, checksum=digest(result)))
        return result


def source_job(source_path, output, fingerprint):
    from .network import UnifiedNetwork
    from .data import ForecastTrace
    source_path = Path(source_path); output = Path(output)
    source_seal = read(source_path / 'complete.json')
    verify(source_seal)
    if source_seal.get('physics_version') != 'native37_complete_plan_ac_hour_start_v4' or source_seal.get('risk_contract') != 'four_boundary_events_complete_plan_v4':
        raise ValueError('Only sealed V4 physical trajectories can enter this offline reuse audit')
    marker = output / 'complete.json'
    if marker.exists():
        saved = read(marker)
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Review code or frozen inputs changed; use a new run-id')
        verify(dict(inputs={}, outputs=saved['outputs']))
        return output.name
    records = read(source_path / 'records.json')
    regulator = CachedRegulator(UnifiedNetwork(), output / 'cache', fingerprint)
    trace = ForecastTrace(); branches = []; details = []
    for i, rec in enumerate(records):
        branch, rows = audit_record(rec, regulator, trace.actual)
        branches.append(branch); details.extend(rows)
        if (i + 1) % 8 == 0:
            print(f'{output.name}: reused {i+1}/{len(records)} branches', flush=True)
    atomic(output / 'branches.json', branches); atomic(output / 'rows.json', details)
    atomic(marker, dict(fingerprint=fingerprint,
                        outputs={str(output / name): sha(output / name) for name in ['branches.json', 'rows.json']},
                        source=str(source_path), branches=len(branches), hours=len(details),
                        new_collection_calls=0, rl_training_calls=0, expert_training_calls=0))
    return output.name


def summarize(root, expected_sources):
    branches = []; rows = []; sources = []
    for marker in sorted((root / 'sources').glob('*/complete.json')):
        verify(dict(inputs={}, outputs=read(marker)['outputs']))
        b = read(marker.parent / 'branches.json'); r = read(marker.parent / 'rows.json')
        branches.extend(b); rows.extend(r)
        y = np.asarray([x['event_new'] for x in b], bool); k = np.asarray([x['known_new'] for x in b], bool)
        sources.append(dict(source_id=marker.parent.name, role=b[0]['role'], branches=len(b),
                            positive=(y & k).sum(0).tolist(), negative=(~y & k).sum(0).tolist(), unknown=(~k).sum(0).tolist(),
                            normal_rows=int((k.all(1) & ~y.any(1)).sum()),
                            complete_compound_rows=int((k.all(1) & (y.sum(1) >= 2)).sum())))
    if not branches:
        return dict(completed_sources=0, expected_sources=expected_sources, complete=False)
    y = np.asarray([b['event_new'] for b in branches], bool); k = np.asarray([b['known_new'] for b in branches], bool)
    old = np.asarray([b['event_old'] for b in branches], bool)
    costs = [r['extra_recourse']['operating_cost_change'] for r in rows if r['extra_recourse']['operating_cost_change'] is not None]
    q_work = [r['normal_regulation']['local_q_change_l1_mvar'] for r in rows if 'local_q_change_l1_mvar' in r['normal_regulation']]
    return dict(completed_sources=len(sources), expected_sources=expected_sources, complete=len(sources) == expected_sources,
                branches=len(branches), hours=len(rows), positive=(y & k).sum(0).tolist(), negative=(~y & k).sum(0).tolist(),
                unknown=(~k).sum(0).tolist(), old_positive=old.sum(0).tolist(),
                old_voltage_positive_to_known_negative=int((old[:,0] & k[:,0] & ~y[:,0]).sum()),
                old_voltage_positive_now_unknown=int((old[:,0] & ~k[:,0]).sum()),
                normal_rows=int((k.all(1) & ~y.any(1)).sum()),
                complete_compound_rows=int((k.all(1) & (y.sum(1) >= 2)).sum()),
                at_least_two_observed_risks=int(((y & k).sum(1) >= 2).sum()),
                normal_Q_status=dict(Counter(r['normal_regulation']['status'] for r in rows)),
                normal_Q_workload=dict(known_hours=len(q_work),
                    local_q_change_l1_mvar_quantiles=np.quantile(q_work,[0,.5,.95,1]).tolist() if q_work else None,
                    summed_branch_hour_effort_mvarh=float(sum(q_work)),
                    count_unit='correlated branch-hours; not independent real-system event incidence'),
                extra_cost_comparable_hours=len(costs),
                extra_cost_change_quantiles=np.quantile(costs, [0, .5, 1]).tolist() if costs else None,
                source_support=sources, new_collection_calls=0, rl_training_calls=0, expert_training_calls=0,
                training_eligible=False, scope='V4-state offline label support audit; not V5 trajectory validation')


def run(source_run, run_id, workers=4, max_sources=0, resume=False):
    source_run = Path(source_run).resolve()
    if not run_id.replace('_','').replace('-','').isalnum():
        raise ValueError('Simple run-id required')
    root = PROJECT / 'artifacts/regulation_reuse' / run_id
    inputs = {str(p): sha(p) for p in (PROJECT / 'src/tarp37').glob('*.py')}
    inputs.update({str(p): sha(p) for p in (PROJECT / 'artifacts/data').glob('*') if p.is_file()})
    markers = sorted((source_run / 'sources').glob('*/complete.json'))
    if not markers:
        raise ValueError('No sealed source data')
    inputs.update({str(p): sha(p) for p in markers})
    for marker in markers:
        seal = read(marker)
        inputs.update(seal['inputs']); inputs.update(seal['outputs'])
    verify(dict(inputs=inputs, outputs={}))
    config = dict(source_run=str(source_run), risk_contract=RISK_VERSION, physics_version=VERSION,
                  sources=[p.parent.name for p in markers], inputs=inputs)
    fingerprint = digest(config)
    if root.exists() and not resume:
        raise ValueError('RunId exists; use --resume')
    if (root / 'audit.json').exists():
        if read(root / 'audit.json') != config:
            raise ValueError('Audit code or frozen inputs changed; use a new run-id')
    else:
        atomic(root / 'audit.json', config)
    selected = markers[:max_sources] if max_sources else markers
    tasks = [(p.parent, root / 'sources' / p.parent.name, fingerprint) for p in selected]
    if workers == 1:
        for task in tasks:
            source_job(*task); atomic(root / 'summary.json', summarize(root, len(markers)))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
            futures = [pool.submit(source_job, *task) for task in tasks]
            for future in as_completed(futures):
                print('sealed ' + future.result(), flush=True)
                atomic(root / 'summary.json', summarize(root, len(markers)))
    # A source label audit cannot mutate source results or its own runtime code.
    verify(dict(inputs=inputs, outputs={}))
    summary = summarize(root, len(markers)); atomic(root / 'summary.json', summary)
    if summary['complete']:
        atomic(root / 'complete.json', dict(fingerprint=fingerprint,
            outputs={str(root / 'summary.json'): sha(root / 'summary.json')},
            new_collection_calls=0, rl_training_calls=0, expert_training_calls=0))
    return dict(root=str(root), **summary)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source-run', required=True); p.add_argument('--run-id', required=True)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--max-sources', type=int, default=0); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); result = run(a.source_run, a.run_id, a.workers, a.max_sources, a.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
