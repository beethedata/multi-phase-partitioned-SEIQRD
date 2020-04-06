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


def get_optimal_parameters(data_path: str, verbose: bool = True) -> Dict[str, Union[float, int]]:
    # Get gt data.
    with open(data_path) as data_file:
        gt_data = json.load(data_file)

    n = gt_data['total_individuals']
    epidemic_evolution = gt_data['epidemic_evolution']
    simulation_duration = len(epidemic_evolution)

    quarantine_start = gt_data.get('quarantine_start', 2 * simulation_duration + 1)
    quarantine_1_duration = gt_data.get('quarantine_1_duration', 2 * simulation_duration + 1)
    quarantine_2_duration = gt_data.get('quarantine_2_duration', 2 * simulation_duration + 1)

    frac_medical = gt_data.get('fraction_medical', 0)
    frac_essential = gt_data.get('fraction_essential', 0)
    frac_others = 1 - frac_medical - frac_essential
    e0_guess = gt_data.get('initial_exposed_guess', 1)
    i0_guess = gt_data.get('initial_infected_guess', 0)
    q0 = gt_data.get('initial_quarantined', 0)
    r0 = gt_data.get('initial_recovered', 0)
    ur0_guess = gt_data.get('initial_unknown_recovered', 0)
    d0 = gt_data.get('initial_deceased', 0)

    gamma_guess = gt_data.get('gamma_guess', 1)
    m_gamma_1_guess = gt_data.get('m_gamma_1_guess', 1)
    m_gamma_reduction_1_guess = gamma_guess / m_gamma_1_guess
    e_gamma_1_guess = gt_data.get('e_gamma_1_guess', 1)
    e_gamma_reduction_1_guess = gamma_guess / e_gamma_1_guess
    o_gamma_1_guess = gt_data.get('o_gamma_1_guess', 1)
    o_gamma_reduction_1_guess = gamma_guess / o_gamma_1_guess
    m_gamma_2_guess = gt_data.get('m_gamma_2_guess', 1)
    m_gamma_reduction_2_guess = m_gamma_1_guess / m_gamma_2_guess
    e_gamma_2_guess = gt_data.get('e_gamma_2_guess', 1)
    e_gamma_reduction_2_guess = e_gamma_1_guess / e_gamma_2_guess
    o_gamma_2_guess = gt_data.get('o_gamma_2_guess', 1)
    o_gamma_reduction_2_guess = o_gamma_1_guess / o_gamma_2_guess
    alpha_guess = gt_data.get('alpha_guess', 0.06)
    delta_guess = gt_data.get('delta_guess', 0)
    sigma_guess = gt_data.get('sigma_guess', 0.744)
    r_i_guess = gt_data.get('r_i_guess', 0.02)
    r_u_guess = gt_data.get('r_u_guess', 0.3)
    r_q_guess = gt_data.get('r_q_guess', 0.05)
    d_i_guess = gt_data.get('d_i_guess', 0.004)
    d_q_guess = gt_data.get('d_q_guess', 0.01)

    def objective_function(pars: List[float]):
        e0, i0, ur0, g, mg_red_1, eg_red_1, og_red_1, mg_red_2, eg_red_2, og_red_2, al, de, s, ri, ru, rq, di, dq = pars
        s0 = n - i0 - e0 - q0 - r0 - d0
        sm0 = frac_medical * s0
        se0 = frac_essential * s0
        so0 = frac_others * s0
        predict_list = solve_ode(sm0=sm0, se0=se0, so0=so0, e0=e0, i0=i0, q0=q0, r0=r0, ur0=ur0, d0=d0,
                                 quarantine_start=quarantine_start,
                                 simulation_duration=simulation_duration,
                                 gamma=g,
                                 m_gamma_reduction_1=mg_red_1, e_gamma_reduction_1=eg_red_1,
                                 o_gamma_reduction_1=og_red_1,
                                 m_gamma_reduction_2=mg_red_2, e_gamma_reduction_2=eg_red_2,
                                 o_gamma_reduction_2=og_red_2,
                                 alpha=al, delta=de, sigma=s,
                                 r_i=ri, r_u=ru, r_q=rq,
                                 d_i=di, d_q=dq)
        return loss_function(predict_list=predict_list, gt_list=epidemic_evolution, total=n)

    initial = [e0_guess, i0_guess, ur0_guess,
               gamma_guess,
               m_gamma_reduction_1_guess, e_gamma_reduction_1_guess, o_gamma_reduction_1_guess,
               m_gamma_reduction_2_guess, e_gamma_reduction_2_guess, o_gamma_reduction_2_guess,
               alpha_guess, delta_guess, sigma_guess,
               r_i_guess, r_u_guess, r_q_guess,
               d_i_guess, d_q_guess]
    bounds = [(0, n), (0, n), (0, n),
              (0, 10),
              (1, 100), (1, 100), (1, 100),
              (1, 100), (1, 100), (1, 100),
              (0, 1), (0, 1), (0, 1),
              (0, 1), (0, 1), (0, 1),
              (0, 1), (0, 1)]

    res = minimize(objective_function, np.array(initial), bounds=bounds, tol=None, options={'disp': True})

    res = res.x
    e0_f, i0_f, ur0_f = res[:3]
    gamma_f = res[3]
    m_gamma_reduction_1_f, e_gamma_reduction_1_f, o_gamma_reduction_1_f = res[4:7]
    m_gamma_reduction_2_f, e_gamma_reduction_2_f, o_gamma_reduction_2_f = res[7:10]
    alpha_f, delta_f, sigma_f = res[10:13]
    r_i_f, r_u_f, r_q_f = res[13:16]
    d_i_f, d_q_f = res[16:18]

    res_dict = {
        'exposed_initial': e0_f,
        'infected_initial': i0_f,
        'unknown_recovered_initial': ur0_f,
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
        'r_u': r_u_f,
        'r_q': r_q_f,
        'd_i': d_i_f,
        'd_q': d_q_f,
    }

    if verbose:
        print('\nThe predicted optimal initial values are:')
        for key, val in res_dict.items():
            print(f'\t{key}:\t{val}')

    s0_f = n - i0_f - e0_f - q0 - r0 - ur0_f - d0
    res_dict['suspected_medical_initial'] = frac_medical * s0_f
    res_dict['suspected_essential_initial'] = frac_essential * s0_f
    res_dict['suspected_others_initial'] = frac_others * s0_f
    res_dict['quarantined_initial'] = q0
    res_dict['recovered_initial'] = r0
    res_dict['deceased_initial'] = d0
    res_dict['quarantine_start'] = quarantine_start
    res_dict['quarantine_1_duration'] = quarantine_1_duration
    res_dict['quarantine_2_duration'] = quarantine_2_duration

    return res_dict


def loss_function(predict_list: List[Dict[str, float]], gt_list: List[Dict[str, float]], total: int) -> float:
    offset = 0
    predict_list = predict_list[offset:]
    gt_list = gt_list[offset:]

    loss = 0
    weight = 1 / 10000
    for i, (predicted_data, gt_data) in enumerate(zip(predict_list, gt_list)):
        for key, gt_val in gt_data.items():
            if key == 'infected':
                continue
            loss += abs(gt_val - predicted_data[key]) * weight
    return loss


if __name__ == '__main__':
    click_get_optimal_parameters()
