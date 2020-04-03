import json
from typing import Dict, List

import click
from scipy.optimize import fmin_slsqp

from ode_solving import solve_ode


@click.command('fit-parameters')
@click.option('--data-path', prompt='Path to json data.')
def get_optimal_parameters(data_path: str, verbose: bool = True) -> Dict[str, float]:
    # Get gt data.
    with open(data_path) as data_file:
        gt_data = json.load(data_file)

    n = gt_data['total_individuals']
    sm0 = int(gt_data['fraction_medical'] * n)
    se0 = int(gt_data['fraction_essential'] * n)
    so0 = n - sm0 - se0
    q0 = gt_data.get('initial_quarantined', 0)
    r0 = gt_data.get('initial_recovered', 0)
    d0 = gt_data.get('initial_deceased', 0)

    quarantine_start = gt_data.get['quarantine_start']
    quarantine_duration = gt_data.get['quarantine_duration']
    epidemic_evolution = gt_data['epidemic_evolution']
    simulation_duration = len(epidemic_evolution)

    e0_guess = gt_data.get('initial_expected_guess', 1)
    i0_guess = gt_data.get('initial_infected_guess', 0)
    gamma_guess = gt_data.get('gamma_guess', 1)
    gamma_m_guess = gt_data.get('gamma_m_guess', 1)
    gamma_e_guess = gt_data.get('gamma_e_guess', 1)
    gamma_o_guess = gt_data.get('gamma_o_guess', 1)
    alpha_guess = gt_data.get('alpha_guess', 0.5)
    delta_guess = gt_data.get('delta_guess', 0)
    sigma_guess = gt_data.get('sigma_guess', 0.9)
    r_i_guess = gt_data.get('r_i_guess', 0.9)
    d_i_guess = gt_data.get('d_i_guess', 0)
    r_q_guess = gt_data.get('r_q_guess', 0.7)
    d_q_guess = gt_data.get('d_q_guess', 0.034)

    def objective_function(params: List[float]):
        e0, i0, gamma, gamma_m, gamma_e, gamma_o, alpha, delta, sigma, r_i, d_i, r_q, d_q = params
        predict_list = solve_ode(sm0=sm0, se0=se0, so0=so0, e0=e0, i0=i0, q0=q0, r0=r0, d0=d0,
                                 quarantine_start=quarantine_start, quarantine_duration=quarantine_duration,
                                 simulation_duration=simulation_duration,
                                 gamma=gamma, gamma_m=gamma_m, gamma_e=gamma_e,
                                 gamma_o=gamma_o, alpha=alpha, delta=delta, sigma=sigma,
                                 r_i=r_i, r_q=r_q, d_i=d_i, d_q=d_q)
        return loss_function(predict_list=predict_list, gt_list=epidemic_evolution)

    initial = [e0_guess, i0_guess,
               gamma_guess, gamma_m_guess, gamma_e_guess, gamma_o_guess,
               alpha_guess, delta_guess, sigma_guess,
               r_i_guess, d_i_guess,
               r_q_guess, d_q_guess]
    bounds = [(0, n), (0, n),
              (0, 10), (0, 10), (0, 10), (0, 10),
              (0, 1), (0, 10), (0, 1),
              (0, 1), (0, 1),
              (0, 1), (0, 1)]

    if verbose:
        result = fmin_slsqp(objective_function, initial, bounds=bounds, iprint=2, iter=1000)
    else:
        result = fmin_slsqp(objective_function, initial, bounds=bounds, iter=1000)

    e0_f, i0_f, gamma_f, gamma_m_f, gamma_e_f, gamma_o_f, alpha_f, delta_f, sigma_f, r_i_f, d_i_f, r_q_f, d_q_f = result
    res_dict = {'expected_initial': e0_f,
                'infected_initial': i0_f,
                'gamma': gamma_f,
                'gamma_m': gamma_m_f,
                'gamma_e': gamma_e_f,
                'gamma_o': gamma_o_f,
                'alpha': alpha_f,
                'delta': delta_f,
                'sigma': sigma_f,
                'r_i': r_i_f,
                'd_i': d_i_f,
                'r_q': r_q_f,
                'd_q': d_q_f}

    return res_dict


def loss_function(predict_list: List[Dict[str, float]], gt_list: List[Dict[str, float]]) -> float:
    squared_diff = 0
    for predicted_data, gt_data in zip(predict_list, gt_list):
        for key, gt_val in gt_data.items():
            squared_diff += (gt_val - predicted_data[key]) ** 2

    return squared_diff
