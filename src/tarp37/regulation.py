"""Routine bounded Q regulation, separate from additional active recourse."""
import numpy as np
import pandapower as pp

from .contract import VERSION


def settle_point(n, result, exo, base_fraction):
    """Same monetary accounting as dispatch; Q has no invented energy price."""
    result = dict(result)
    shed = float(exo[0]) * (1. - base_fraction)
    wc = max(float(exo[1]) - result['wind_used_mw'], 0.)
    sc = max(float(exo[2]) - result['solar_used_mw'], 0.)
    generation = 0.
    for g in result['generators']:
        a, b, z = n.costs[g['id']]
        factor = max(float(exo[3]), 1.) / 30. if g['bus'] in n.roots else 1.
        generation += factor * (a * g['p_mw'] ** 2 + b * g['p_mw'] + z)
    recovery = shed * n.shed_unit_cost + (wc + sc) * n.curtail_unit_cost + result['backup_mwh'] * n.backup_unit_cost
    result.update(status='routine_Q_ac_valid', scope='bounded_routine_Q_after_normal_regulation', physics_version=VERSION,
        base_fraction=base_fraction, non_ev_shed_mwh=shed,
        non_ev_shed_by_bus_mw=(n.load_share * shed).tolist(),
        wind_curtail_mwh=wc, solar_curtail_mwh=sc,
        wind_available_mwh=float(exo[1]), solar_available_mwh=float(exo[2]),
        generation_cost=float(generation), recovery_cost=float(recovery),
        total_operating_cost=float(generation + recovery),
        ac_loss_regularization_cost=float(n.ac_loss_unit_value * result['loss_mwh']),
        dispatch_objective_cost=float(generation + recovery + n.ac_loss_unit_value * result['loss_mwh']),
        protection_attempts=[], protection_policy='routine_bounded_Q_before_additional_AC_recourse')
    return result


class RoutineQRegulator:
    def __init__(self, network):
        self.n = network

    def _solve(self, exo, power, point):
        n = self.n
        net, gens, re, back, base, q, dev = n._build(
            exo, power, point['base_fraction'], point['backup_mw'], False, point)
        for bus in net.bus.index:
            root = n.bus_ids[bus] in n.roots
            net.bus.at[bus, 'min_vm_pu'] = 1. if root else n.c.v_min
            net.bus.at[bus, 'max_vm_pu'] = 1. if root else n.c.v_max
        net.line['max_loading_percent'] = 100.
        gen_indices = set()
        for g, kind, idx in gens:
            if kind == 'sgen':
                gen_indices.add(idx)
                pg = float(net.sgen.at[idx, 'p_mw'])
                net.sgen.at[idx, 'controllable'] = True
                net.sgen.at[idx, 'min_p_mw'] = pg
                net.sgen.at[idx, 'max_p_mw'] = pg
            pp.create_poly_cost(net, idx, kind, cp1_eur_per_mw=0., cp2_eur_per_mw2=1., cq2_eur_per_mvar2=.01)
        for idx in net.sgen.index:
            if idx in gen_indices:
                continue
            pg = float(net.sgen.at[idx, 'p_mw'])
            net.sgen.at[idx, 'controllable'] = True
            net.sgen.at[idx, 'min_p_mw'] = pg
            net.sgen.at[idx, 'max_p_mw'] = pg
        pp.runopp(net, verbose=False, init='flat', numba=False, delta=1e-7,
                  OPF_VIOLATION=1e-7, OPF_FLOW_LIM=2, PDIPM_MAX_IT=200)
        for idx in net.sgen.index:
            net.sgen.at[idx, 'p_mw'] = float(net.sgen.at[idx, 'min_p_mw'])
            net.sgen.at[idx, 'q_mvar'] = float(np.clip(net.res_sgen.at[idx, 'q_mvar'],
                net.sgen.at[idx, 'min_q_mvar'], net.sgen.at[idx, 'max_q_mvar']))
        pp.runpp(net, init='results', numba=False, max_iteration=100,
                 tolerance_mva=1e-10, calculate_voltage_angles=True)
        result = n._read(net, gens, re, back, base, q, dev)
        for j, g in enumerate(n.gen):
            if g['bus'] not in n.roots and abs(result['generators'][j]['p_mw'] - point['generation_mw'][j]) > 1e-10:
                raise RuntimeError('Normal Q controller changed local active generation')
        result['known'] = result['max_bus_residual_mva'] <= 1e-5
        result['base_fraction'] = point['base_fraction']
        return result

    def _q_valid(self, result):
        n = self.n
        return all(g['qmin_mvar'] - n.c.power_tolerance_mw <= out['q_mvar'] <= g['qmax_mvar'] + n.c.power_tolerance_mw
                   for g, out in zip(n.gen, result.get('generators', [])))

    def review(self, exo, power, routine):
        n = self.n; pre = n.pressure(exo, power, routine)
        if not pre.get('known'):
            return dict(known=False, physical_valid=False, point=None,
                        status='initial_AC_unknown', residual_voltage_event=None)
        if pre['voltage_excess_pu'] <= n.c.voltage_tolerance_pu and self._q_valid(pre):
            result = pre
            status = 'already_regulated' if pre['physical_valid'] else 'voltage_ok_other_boundary'
        else:
            if pre.get('pcc_export_permission_excess_mw', 0.) > n.c.power_tolerance_mw:
                return dict(known=False, physical_valid=False, point=None,
                            status='active_export_boundary_requires_separate_recours', residual_voltage_event=None)
            try:
                result = self._solve(exo, power, routine)
            except (pp.OPFNotConverged, pp.LoadflowNotConverged) as err:
                return dict(known=False, physical_valid=False, point=None,
                            status='normal_Q_numerical_unknown', error=type(err).__name__,
                            residual_voltage_event=None)
            if not result.get('known') or not self._q_valid(result):
                return dict(known=False, physical_valid=False, point=None,
                            status='normal_Q_independent_check_unknown', residual_voltage_event=None)
            status = 'Q_corrected' if result['physical_valid'] else 'Q_residual_constraint'
        if result['physical_valid']:
            result = settle_point(n, result, exo, float(routine['base_fraction']))
        local = [j for j, g in enumerate(n.gen) if g['bus'] not in n.roots]
        delta = np.array([g['q_mvar'] for g in result['generators']]) - np.asarray(routine['q_mvar'])
        return dict(known=True, physical_valid=bool(result['physical_valid']), point=result, status=status,
                    residual_voltage_event=bool(result['voltage_excess_pu'] > n.c.voltage_tolerance_pu),
                    q_change_mvar=delta.tolist(), local_q_change_l1_mvar=float(np.abs(delta[local]).sum()),
                    local_active_commands_preserved=True, necessity_of_extra_active_recours_proven=False)


def extra_recourse(normal, actual):
    comparable = bool(normal.get('physical_valid') and actual.get('physical_valid'))
    out = dict(normal_baseline_valid=comparable,
               actual_operating_cost=actual.get('total_operating_cost'),
               actual_generation_cost=actual.get('generation_cost'),
               actual_recovery_cost=actual.get('recovery_cost'),
               actual_recovery_components={k:actual.get(k) for k in ['wind_curtail_mwh','solar_curtail_mwh','backup_mwh','non_ev_shed_mwh']},
               operating_cost_change=None,
               interpretation='signed observed change relative to valid normal point; not voltage-only necessity')
    if not comparable:
        return out
    baseline = normal['point']
    out['normal_operating_cost'] = baseline['total_operating_cost']
    out['operating_cost_change'] = actual['total_operating_cost'] - baseline['total_operating_cost']
    for field in ['wind_curtail_mwh', 'solar_curtail_mwh', 'backup_mwh', 'non_ev_shed_mwh']:
        delta = actual[field] - baseline[field]
        out[field + '_change'] = delta
        out['additional_' + field] = max(delta, 0.)
    out['local_and_root_generation_change_mw'] = [a['p_mw'] - b['p_mw'] for a, b in zip(actual['generators'], baseline['generators'])]
    out['q_change_mvar'] = [a['q_mvar'] - b['q_mvar'] for a, b in zip(actual['generators'], baseline['generators'])]
    return out
