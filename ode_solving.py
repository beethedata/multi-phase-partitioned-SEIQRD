from typing import Dict, List

from scipy.integrate import ode
import click

from visualization import show_results


# Define the ordinary differential equation we must solve in order to compute epidemic evolution.
def diff_equations(t, y, params):
    gamma, gamma_m, gamma_e, gamma_o, alpha, delta, sigma, r_i, r_q, d_i, d_q, q_s, q_d = params
    sm, se, so, e, i, q, r, d = y
    n = sm + se + so + e + i + q + r + d

    sm_dt = -sm * ((gamma_m if q_s <= t < q_s + q_d else gamma) * i + delta * q) / n
    se_dt = -se * ((gamma_e if q_s <= t < q_s + q_d else gamma) * i) / n
    so_dt = -so * ((gamma_e if q_s <= t < q_s + q_d else gamma) * i) / n
    e_dt = -(sm_dt + se_dt + so_dt) - sigma * e
    i_dt = sigma * e - (alpha + r_i + d_i) * i
    q_dt = alpha * i - (r_q + d_q) * q
    r_dt = r_i * i + r_q * q
    d_dt = d_i * i + d_q * q

    return [sm_dt, se_dt, so_dt, e_dt, i_dt, q_dt, r_dt, d_dt]


@click.command('run-model')
@click.option('--sm0', prompt='SM0',
              help='Number of susceptible individuals in the `medical` sector at day 0 of simulation')
@click.option('--se0', prompt='SE0',
              help='Number of susceptible individuals in the `essential services` sector at day 0 of simulation')
@click.option('--so0', prompt='SO0',
              help='Number of susceptible individuals in the `others` sector at day 0 of simulation.')
@click.option('--e0', prompt='E0', help='Number of exposed individuals at day 0 of simulation.')
@click.option('--i0', prompt='I0', help='Number of infected individuals at day 0 of simulation.')
@click.option('--q0', prompt='Q0', help='Number of hospitalized individuals at day 0 of simulation.')
@click.option('--r0', prompt='R0', help='Number of recovered individuals at day 0 of simulation.')
@click.option('--d0', prompt='D0', help='Number of deceased individuals at day 0 of simulation.')
@click.option('--gamma', prompt='gamma', help='Propagation factor before quarantine is imposed.')
@click.option('--gamma-m', prompt='gamma medical',
              help='Propagation factor for medical sector after quarantine is imposed.')
@click.option('--gamma-e', prompt='gamma essential services',
              help='Propagation factor for essential services sector after quarantine is imposed.')
@click.option('--gamma-o', prompt='gamma others',
              help='Propagation factor for other sector after quarantine is imposed.')
@click.option('--alpha', prompt='alpha', help='Probability of being hospitalized if infectious.')
@click.option('--delta', prompt='delta', help='Propagation factor from hospitalized to medical sector.')
@click.option('--sigma', prompt='sigma', help='Probability of exposed individual to become infected.')
@click.option('--quarantine-start', prompt='quarantine start day',
              help='Integer indicating what day since day 0 quarantine is imposed.')
@click.option('--quarantine-duration', prompt='Duration of quarantine (days)',
              help='Integer indicating how many days will the quarantine last.')
@click.option('--quarantine-duration', prompt='Duration of quarantine (days)',
              help='Integer indicating how many days will the quarantine last.')
def run_model(sm0: int, se0: int, so0: int, e0: int = 1, i0: int = 0, q0: int = 0, r0: int = 0, d0: int = 0,
              quarantine_start: int = 25, quarantine_duration: int = 40, simulation_duration: int = 100,
              gamma: float = 1, gamma_m: float = 1, gamma_e: float = 1, gamma_o: float = 1,
              alpha: float = 0.5, delta: float = 0, sigma: float = 0.9,
              r_i: float = 0.9, r_q: float = 0.7,
              d_i: float = 0, d_q: float = 0.034) -> List[Dict[str, float]]:
    result_list = solve_ode(sm0=sm0, se0=se0, so0=so0, e0=e0, i0=i0, q0=q0, r0=r0, d0=d0,
                            quarantine_start=quarantine_start, quarantine_duration=quarantine_duration,
                            simulation_duration=simulation_duration,
                            gamma=gamma, gamma_m=gamma_m, gamma_e=gamma_e, gamma_o=gamma_o,
                            alpha=alpha, delta=delta, sigma=sigma,
                            r_i=r_i, r_q=r_q, d_i=d_i, d_q=d_q)

    show_results(result_list)

    return result_list


def solve_ode(sm0: int, se0: int, so0: int, e0: int, i0: int = 0, q0: int = 0, r0: int = 0, d0: int = 0,
              quarantine_start: int = 25, quarantine_duration: int = 40, simulation_duration: int = 100,
              gamma: float = 1, gamma_m: float = 1, gamma_e: float = 1, gamma_o: float = 1,
              alpha: float = 0.5, delta: float = 0, sigma: float = 0.9,
              r_i: float = 0.9, r_q: float = 0.7,
              d_i: float = 0, d_q: float = 0.034) -> List[Dict[str, float]]:
    # Initialize an object to solve the differential equation.
    ode_solver = ode(diff_equations).set_integrator('dopri5', nsteps=10000)

    # Set initial value.
    ode_solver.set_initial_value([sm0, se0, so0, e0, i0, q0, r0, d0], 0)

    # Set parameters.
    ode_solver.set_f_params([gamma, gamma_m, gamma_e, gamma_o, alpha, delta, sigma,
                             r_i, r_q, d_i, d_q,
                             quarantine_start, quarantine_duration])
    result_list: List[Dict[str, float]] = [
        {
            'susceptible': sm0 + se0 + so0,
            'exposed': e0,
            'infected': i0,
            'quarantined': q0,
            'recovered': r0,
            'deceased': d0,
        }
    ]
    while ode_solver.successful() and ode_solver.t <= simulation_duration:
        ode_solver.integrate(ode_solver.t + 1)
        sm, se, so, e, i, q, r, d = ode_solver.y
        result_list.append(
            {
                'susceptible_medical': sm,
                'susceptible_essential_services': se,
                'susceptible_others': so,
                'susceptible': sm + se + so,
                'exposed': e,
                'infected': i,
                'quarantined': q,
                'recovered': r,
                'deceased': d,
            }
        )

    return result_list
