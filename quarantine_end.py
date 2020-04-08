import json
from typing import Optional

import click

from ode_solving import solve_ode
from visualization import show_multiple_results


@click.command('quarantine-end-prediction')
@click.option('--parameters-path', prompt='Parameters path', help='path where model parameters are stored.',
              required=True)
@click.option('--simulation-duration', prompt='Simulation duration', help='Duration of simulation.', default=100)
@click.option('--offset', prompt='Offset from day 0', help='Offset from day 0 when showing.', default=0)
@click.option('--top-lim', help='Offset from day 0 when showing.', type=int)
def simulate_quarantine_end(parameters_path: str, simulation_duration: int = 100, offset: int = 0,
                            top_lim: Optional[int] = None):
    with open(parameters_path) as data_file:
        parameters = json.load(data_file)

    result_list = []
    for quarantine_2_duration, date in parameters["quarantine_2_duration_list"]:
        results = solve_ode(sm0=parameters["suspected_medical_initial"],
                            se0=parameters["suspected_essential_initial"],
                            so0=parameters["suspected_others_initial"],
                            e0=parameters["exposed_initial"],
                            i0=parameters["infected_initial"],
                            q0=parameters["quarantined_initial"],
                            r0=parameters["recovered_initial"],
                            ur0=parameters["unknown_recovered_initial"],
                            d0=parameters["deceased_initial"],
                            quarantine_start=parameters["quarantine_start"],
                            quarantine_1_duration=parameters["quarantine_1_duration"],
                            quarantine_2_duration=quarantine_2_duration,
                            simulation_duration=simulation_duration,
                            gamma=parameters["gamma"],
                            m_gamma_reduction_1=parameters["gamma"] / parameters["gamma_m_1"],
                            e_gamma_reduction_1=parameters["gamma"] / parameters["gamma_e_1"],
                            o_gamma_reduction_1=parameters["gamma"] / parameters["gamma_o_1"],
                            m_gamma_reduction_2=parameters["gamma_m_1"] / parameters["gamma_m_2"],
                            e_gamma_reduction_2=parameters["gamma_e_1"] / parameters["gamma_e_2"],
                            o_gamma_reduction_2=parameters["gamma_o_1"] / parameters["gamma_o_2"],
                            alpha=parameters["alpha"],
                            delta=parameters["delta"],
                            sigma=parameters["sigma"],
                            r_i=parameters["r_i"],
                            r_u=parameters["r_u"],
                            r_q=parameters["r_q"],
                            d_i=parameters["d_i"],
                            d_q=parameters["d_q"])
        end_day = parameters["quarantine_start"] + parameters["quarantine_1_duration"] + quarantine_2_duration
        result_list.append((results, date, end_day))

    show_multiple_results(result_list, title='Infected depending on quarantine end', offset=offset, top_lim=top_lim,
                          gt_data=parameters.get('gt_data', None))


if __name__ == '__main__':
    simulate_quarantine_end()
