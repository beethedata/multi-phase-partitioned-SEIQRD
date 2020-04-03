import click

from model_fitting import get_optimal_parameters
from ode_solving import run_model


@click.group()
def cli():
    pass


cli.add_command(get_optimal_parameters)
cli.add_command(run_model)
