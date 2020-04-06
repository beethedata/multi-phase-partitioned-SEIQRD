import json
from typing import Optional

import click

from model_fitting import click_get_optimal_parameters, get_optimal_parameters
from ode_solving import run_model
from quarantine_end import simulate_quarantine_end


@click.group()
def cli():
    pass


@click.command('fit-and-predict')
@click.option('--train-data-path', prompt='Path to json data used for training', required=True)
@click.option('--gt-data-path', prompt='Path to json with ground truth data', default=None)
@click.option('--simulation-duration', default=None)
def fit_and_predict(train_data_path: str, gt_data_path: Optional[str] = None, simulation_duration: Optional[int] = None):
    gt_data = None
    if gt_data_path is not None:
        with open(gt_data_path) as data_file:
            gt_data = json.load(data_file)
            gt_data = gt_data['epidemic_evolution']

    if simulation_duration is None:
        assert gt_data is not None
        simulation_duration = len(gt_data) - 1

    res_dict = get_optimal_parameters(data_path=train_data_path, verbose=True)
    run_model(sm0=res_dict['suspected_medical_initial'],
              se0=res_dict['suspected_essential_initial'],
              so0=res_dict['suspected_others_initial'],
              e0=res_dict['exposed_initial'],
              i0=res_dict['infected_initial'],
              q0=res_dict['quarantined_initial'],
              r0=res_dict['recovered_initial'],
              ur0=res_dict['unknown_recovered_initial'],
              d0=res_dict['deceased_initial'],
              quarantine_start=res_dict['quarantine_start'],
              quarantine_1_duration=res_dict['quarantine_1_duration'],
              quarantine_2_duration=res_dict['quarantine_2_duration'],
              simulation_duration=simulation_duration,
              gamma=res_dict['gamma'],
              gamma_m_1=res_dict['gamma_m_1'],
              gamma_e_1=res_dict['gamma_e_1'],
              gamma_o_1=res_dict['gamma_o_1'],
              gamma_m_2=res_dict['gamma_m_2'],
              gamma_e_2=res_dict['gamma_e_2'],
              gamma_o_2=res_dict['gamma_o_2'],
              alpha=res_dict['alpha'],
              delta=res_dict['delta'],
              sigma=res_dict['sigma'],
              r_i=res_dict['r_i'],
              r_u=res_dict['r_u'],
              r_q=res_dict['r_q'],
              d_i=res_dict['d_i'],
              d_q=res_dict['d_q'],
              gt_data=gt_data)


cli.add_command(fit_and_predict)
cli.add_command(click_get_optimal_parameters)
cli.add_command(simulate_quarantine_end)


if __name__ == '__main__':
    cli()
