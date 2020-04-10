import json
from typing import Dict, List, Union

import click
import numpy as np
from scipy.optimize import minimize

from ode_solving import solve_ode


@click.command('fit-parameters')
@click.option('--data-path', prompt='Path to json data.')
def click_get_optimal_parameters(data_path: str):
    return get_optimal_parameters(data_path=data_path, verbose=True)


def get_optimal_parameters(data_path: str, method: str = '', verbose: bool = True,
                           total: int = 10000) -> Dict[str, Union[float, int]]:
    # Get gt data.
    with open(data_path) as data_file:
        gt_data = json.load(data_file)

    n = gt_data['total_individuals']
    epidemic_evolution = gt_data['epidemic_evolution']
    simulation_duration = len(epidemic_evolution)
    assert simulation_duration > 0
    q0 = epidemic_evolution[0].get('quarantined', 0) * total / n
    r0 = epidemic_evolution[0].get('recovered', 0) * total / n
    d0 = epidemic_evolution[0].get('deceased', 0) * total / n

    quarantine_start = gt_data.get('quarantine_start', 2 * simulation_duration + 1)
    quarantine_1_duration = gt_data.get('quarantine_1_duration', 2 * simulation_duration + 1)
    quarantine_2_duration = gt_data.get('quarantine_2_duration', 2 * simulation_duration + 1)

    frac_medical = gt_data.get('fraction_medical', 0)
    frac_essential = gt_data.get('fraction_essential', 0)
    frac_others = 1 - frac_medical - frac_essential
    e0_guess = gt_data.get('initial_exposed_guess', 1) * total / n
    i0_guess = gt_data.get('initial_infected_guess', 1) * total / n

    epsilon = 0
    default_gamma = 1.96
    default_gamma_reduction_1 = 1
    default_gamma_reduction_2 = 1
    gamma_guess = gt_data.get('gamma_guess', default_gamma)
    m_gamma_1_guess = gt_data.get('m_gamma_1_guess', gamma_guess / default_gamma_reduction_1)
    m_gamma_reduction_1_guess = max(gamma_guess / m_gamma_1_guess, 1 + epsilon)
    e_gamma_1_guess = gt_data.get('e_gamma_1_guess', gamma_guess / default_gamma_reduction_1)
    e_gamma_reduction_1_guess = max(gamma_guess / e_gamma_1_guess, 1 + epsilon)
    o_gamma_1_guess = gt_data.get('o_gamma_1_guess', gamma_guess / default_gamma_reduction_1)
    o_gamma_reduction_1_guess = max(gamma_guess / o_gamma_1_guess, 1 + epsilon)
    m_gamma_2_guess = gt_data.get('m_gamma_2_guess', m_gamma_1_guess / default_gamma_reduction_2)
    m_gamma_reduction_2_guess = max(m_gamma_1_guess / m_gamma_2_guess, 1 + epsilon)
    e_gamma_2_guess = gt_data.get('e_gamma_2_guess', e_gamma_1_guess / default_gamma_reduction_2)
    e_gamma_reduction_2_guess = max(e_gamma_1_guess / e_gamma_2_guess, 1 + epsilon)
    o_gamma_2_guess = gt_data.get('o_gamma_2_guess', o_gamma_1_guess / default_gamma_reduction_2)
    o_gamma_reduction_2_guess = max(o_gamma_1_guess / o_gamma_2_guess, 1 + epsilon)
    alpha_guess = gt_data.get('alpha_guess', 0.2)
    delta_guess = max(gt_data.get('delta_guess', 0), epsilon)
    sigma_guess = gt_data.get('sigma_guess', 0.196)
    r_i_guess = gt_data.get('r_i_guess', 0.222)
    r_q_guess = gt_data.get('r_q_guess', 0.222)
    d_i_guess = gt_data.get('d_i_guess', 0.053)
    d_q_guess = gt_data.get('d_q_guess', 0.053)

    def objective_function(pars: List[float]):
        e0, i0, g, mg_red_1, eg_red_1, og_red_1, mg_red_2, eg_red_2, og_red_2, al, de, s, ri, rq, di, dq = pars
        s0 = total - i0 - e0 - q0 - r0 - d0
        sm0 = frac_medical * s0
        se0 = frac_essential * s0
        so0 = frac_others * s0
        predict_list = solve_ode(sm0=sm0, se0=se0, so0=so0, e0=e0, i0=i0, q0=q0, r0=r0, d0=d0,
                                 quarantine_start=quarantine_start,
                                 quarantine_1_duration=quarantine_1_duration,
                                 quarantine_2_duration=quarantine_2_duration,
                                 simulation_duration=simulation_duration,
                                 gamma=g,
                                 m_gamma_reduction_1=mg_red_1, e_gamma_reduction_1=eg_red_1,
                                 o_gamma_reduction_1=og_red_1,
                                 m_gamma_reduction_2=mg_red_2, e_gamma_reduction_2=eg_red_2,
                                 o_gamma_reduction_2=og_red_2,
                                 alpha=al, delta=de, sigma=s,
                                 r_i=ri, r_q=rq,
                                 d_i=di, d_q=dq)
        return loss_function(predict_list=predict_list, gt_list=epidemic_evolution, fictional_total=total, total=n)

    initial = [e0_guess, i0_guess,
               gamma_guess,
               m_gamma_reduction_1_guess, e_gamma_reduction_1_guess, o_gamma_reduction_1_guess,
               m_gamma_reduction_2_guess, e_gamma_reduction_2_guess, o_gamma_reduction_2_guess,
               alpha_guess, delta_guess, sigma_guess,
               r_i_guess, r_q_guess,
               d_i_guess, d_q_guess]
    bounds = [(0, total), (0, total),
              (0, 10),
              (1, 100), (1, 100), (1, 100),
              (1, 100), (1, 100), (1, 100),
              (0, 1), (0, 1), (0, 1),
              (0, 1), (0, 1),
              (0, 1), (0, 1)]

    # method = 'nelder-mead'  # not bad.  decrease fast.
    # method = 'powell'
    # method = 'cg'
    # method = 'bfgs'
    # method = 'l-bfgs-b'  # not bad. very linear.  # try reduction of 1.
    # method = 'tnc'  # try incentivating end.
    # method = 'cobyla'
    # method = 'slsqp'
    method = 'trust-constr'  # like.
    # method = None
    res = minimize(objective_function, np.array(initial), bounds=bounds, method=method,
                   tol=None, options={'disp': True})

    res = res.x
    e0_f, i0_f = res[:2]
    gamma_f = res[2]
    m_gamma_reduction_1_f, e_gamma_reduction_1_f, o_gamma_reduction_1_f = res[3:6]
    m_gamma_reduction_2_f, e_gamma_reduction_2_f, o_gamma_reduction_2_f = res[6:9]
    alpha_f, delta_f, sigma_f = res[9:12]
    r_i_f, r_q_f = res[12:14]
    d_i_f, d_q_f = res[14:16]

    res_dict = {
        'exposed_initial': e0_f * n / total,
        'infected_initial': i0_f * n / total,
        'gamma': gamma_f,
        'gamma_m_1': gamma_f / m_gamma_reduction_1_f,
        'gamma_e_1': gamma_f / e_gamma_reduction_1_f,
        'gamma_o_1': gamma_f / o_gamma_reduction_1_f,
        'gamma_m_2': gamma_f / m_gamma_reduction_1_f / m_gamma_reduction_2_f,
        'gamma_e_2': gamma_f / e_gamma_reduction_1_f / e_gamma_reduction_2_f,
        'gamma_o_2': gamma_f / o_gamma_reduction_1_f / o_gamma_reduction_2_f,
        'alpha': alpha_f,
        'delta': delta_f,
        'sigma': sigma_f,
        'r_i': r_i_f,
        'r_q': r_q_f,
        'd_i': d_i_f,
        'd_q': d_q_f,
    }

    if verbose:
        print('\nThe predicted optimal initial values are:')
        for key, val in res_dict.items():
            print(f'\t{key}:\t{val}')

    s0_f = total - i0_f - e0_f - q0 - r0 - d0
    res_dict['suspected_medical_initial'] = frac_medical * s0_f * n / total
    res_dict['suspected_essential_initial'] = frac_essential * s0_f * n / total
    res_dict['suspected_others_initial'] = frac_others * s0_f * n / total
    res_dict['quarantined_initial'] = q0 * n / total
    res_dict['recovered_initial'] = r0 * n / total
    res_dict['deceased_initial'] = d0 * n / total
    res_dict['quarantine_start'] = quarantine_start
    res_dict['quarantine_1_duration'] = quarantine_1_duration
    res_dict['quarantine_2_duration'] = quarantine_2_duration

    return res_dict


def loss_function(predict_list: List[Dict[str, float]], gt_list: List[Dict[str, float]],
                  fictional_total: int, total: int,
                  weight: float = 1, deceased_relative_weight: float = 10, offset: int = 0) -> float:
    predict_list = predict_list[offset:]
    gt_list = gt_list[offset:]

    quarantined_loss = 0
    deceased_loss = 0
    quarantined_weight = weight
    n_quarantined = 0
    n_deceased = 0
    deceased_weight = weight * deceased_relative_weight
    n = min(len(predict_list), len(gt_list))
    for i, (predicted_data, gt_data) in enumerate(zip(predict_list, gt_list)):
        # step_weight = ((i + 1) / n) ** 2
        # step_weight = (i + 1) / n
        step_weight = 1
        for key, gt_val in gt_data.items():
            gt_val = gt_val * fictional_total / total
            if 'quarantined' in key:
                quarantined_loss += abs(gt_val - predicted_data[key]) * quarantined_weight * step_weight
                n_quarantined += 1
            elif 'deceased' in key:
                deceased_loss += abs(gt_val - predicted_data[key]) * deceased_weight * step_weight
                n_deceased += 1
    if n_quarantined > 0:
        quarantined_loss /= n_quarantined
    if n_deceased > 0:
        deceased_loss /= n_deceased
    # print(f'\nq_loss:\t{quarantined_loss:.3f}\nd_loss:\t{deceased_loss:.3f}')
    return quarantined_loss + deceased_loss


if __name__ == '__main__':
    click_get_optimal_parameters()
