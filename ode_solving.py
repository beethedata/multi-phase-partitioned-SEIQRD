import json
from typing import Dict, List, Optional

from scipy.integrate import ode
import click

from visualization import show_results


# Define the ordinary differential equation we must solve in order to compute epidemic evolution.
def diff_equations(t, y, par):
    g, mg_1, eg_1, og_1, mg_2, eg_2, og_2, al, de, s, ri, ru, rq, di, dq, start, dur_1, dur_2 = par
    sm, se, so, e, i, q, r, ur, d = y
    n = sm + se + so + e + i + q + r + ur + d

    alpha = al
    delta = d
    sigma = s
    if start <= t < start + dur_1:
        gamma_m = g / mg_1
        gamma_e = g / eg_1
        gamma_o = g / og_1
    elif start + dur_1 <= t < start + dur_1 + dur_2:
        gamma_m = g / mg_1 / mg_2
        gamma_e = g / eg_1 / eg_2
        gamma_o = g / og_1 / og_2
    else:
        gamma_m = g
        gamma_e = g
        gamma_o = g

    sm_dt = -sm * (gamma_m * i + delta * q) / n
    se_dt = -se * (gamma_e * i) / n
    so_dt = -so * (gamma_o * i) / n
    e_dt = -(sm_dt + se_dt + so_dt) - sigma * e
    i_dt = sigma * e - (alpha + ri + di + ru) * i
    q_dt = alpha * i - (rq + dq) * q
    r_dt = ri * i + rq * q
    ur_dt = ru * i
    d_dt = di * i + dq * q

    return [sm_dt, se_dt, so_dt, e_dt, i_dt, q_dt, r_dt, ur_dt, d_dt]


def run_model(sm0: int, se0: int, so0: int, e0: int = 1, i0: int = 0, q0: int = 0,
              r0: int = 0, ur0: int = 0, d0: int = 0,
              quarantine_start: int = 25,
              quarantine_1_duration: int = 14, quarantine_2_duration: int = 40,
              simulation_duration: int = 1000,
              gamma: float = 1,
              gamma_m_1: float = 1, gamma_e_1: float = 1, gamma_o_1: float = 1,
              gamma_m_2: float = 1, gamma_e_2: float = 1, gamma_o_2: float = 1,
              alpha: float = 0.5, delta: float = 0, sigma: float = 0.9,
              r_i: float = 0.9, r_u: float = 0.9, r_q: float = 0.7,
              d_i: float = 0, d_q: float = 0.034,
              gt_data: Optional[List[Dict[str, float]]] = None) -> List[Dict[str, float]]:
    gt_data = gt_data[:simulation_duration + 1]
    result_list = solve_ode(sm0=sm0, se0=se0, so0=so0, e0=e0, i0=i0, q0=q0, r0=r0, ur0=ur0, d0=d0,
                            quarantine_start=quarantine_start,
                            quarantine_1_duration=quarantine_1_duration, quarantine_2_duration=quarantine_2_duration,
                            simulation_duration=simulation_duration,
                            gamma=gamma,
                            m_gamma_reduction_1=gamma / gamma_m_1,
                            e_gamma_reduction_1=gamma / gamma_e_1,
                            o_gamma_reduction_1=gamma / gamma_o_1,
                            m_gamma_reduction_2=gamma_m_1 / gamma_m_2,
                            e_gamma_reduction_2=gamma_e_1 / gamma_e_2,
                            o_gamma_reduction_2=gamma_o_1 / gamma_o_2,
                            alpha=alpha, delta=delta, sigma=sigma,
                            r_i=r_i, r_u=r_u, r_q=r_q,
                            d_i=d_i, d_q=d_q)

    show_results(result_list, gt_data=gt_data)

    return result_list


def solve_ode(sm0: int, se0: int, so0: int, e0: int, i0: int = 0, q0: int = 0, r0: int = 0, ur0: int = 0, d0: int = 0,
              quarantine_start: int = 25,
              quarantine_1_duration: int = 14,
              quarantine_2_duration: int = 1000,
              simulation_duration: int = 100,
              gamma: float = 1,
              m_gamma_reduction_1: float = 1, e_gamma_reduction_1: float = 1, o_gamma_reduction_1: float = 1,
              m_gamma_reduction_2: float = 1, e_gamma_reduction_2: float = 1, o_gamma_reduction_2: float = 1,
              alpha: float = 0.5, delta: float = 0, sigma: float = 0.9,
              r_i: float = 0.9, r_u: float = 0.9, r_q: float = 0.7,
              d_i: float = 0, d_q: float = 0.034) -> List[Dict[str, float]]:
    n = sm0 + se0 + so0 + e0 + i0 + q0 + r0 + ur0 + d0

    # Initialize an object to solve the differential equation.
    ode_solver = ode(diff_equations).set_integrator('dopri5', nsteps=10000)

    # Set initial value.
    ode_solver.set_initial_value([sm0 / n, se0 / n, so0 / n, e0 / n, i0 / n, q0 / n, r0 / n, ur0 / n, d0 / n], 0)

    # Set parameters.
    ode_solver.set_f_params([gamma,
                             m_gamma_reduction_1, e_gamma_reduction_1, o_gamma_reduction_1,
                             m_gamma_reduction_2, e_gamma_reduction_2, o_gamma_reduction_2,
                             alpha, delta, sigma,
                             r_i, r_u, r_q,
                             d_i, d_q,
                             quarantine_start, quarantine_1_duration, quarantine_2_duration])
    result_list: List[Dict[str, float]] = [
        {
            'susceptible_medical': sm0,
            'susceptible_essential_services': se0,
            'susceptible_others': so0,
            'susceptible': sm0 + se0 + so0,
            'exposed': e0,
            'infected': i0,
            'quarantined': q0,
            'recovered': r0,
            'unknown_recovered': ur0,
            'deceased': d0,
        }
    ]
    step = 1
    t = step
    while ode_solver.successful() and t <= simulation_duration:
        ode_solver.integrate(t)
        t += step
        sm, se, so, e, i, q, r, ur, d = ode_solver.y
        result_list.append(
            {
                'susceptible_medical': sm * n,
                'susceptible_essential_services': se * n,
                'susceptible_others': so * n,
                'susceptible': (sm + se + so) * n,
                'exposed': e * n,
                'infected': i * n,
                'quarantined': q * n,
                'recovered': r * n,
                'unknown_recovered': ur * n,
                'deceased': d * n,
            }
        )

    return result_list
