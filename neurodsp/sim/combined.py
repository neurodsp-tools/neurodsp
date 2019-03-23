"""Simulating time series, with combinations of periodic, aperiodic and transient components."""

from itertools import repeat

from neurodsp.sim.info import get_sim_func
from neurodsp.sim.decorators import normalize
from neurodsp.sim.utils import proportional_sum

###################################################################################################
###################################################################################################

@normalize
def sim_combined(n_seconds, fs, simulations, proportions=1):
    """Sim multiple component signals and combine them.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    simulations : dictionary
        A dictionary of simulation functions to run, with their desired parameters.
    proportions : float of list of float
        Variance proportions to use to sum across signals.

    Returns
    -------
    sig : 1d array
        Simulated combined signal.
    """

    if not (proportions == 1 or len(proportions) == len(simulations)):
        raise ValueError('Simulations and proportions lengths do not match.')

    simulations = {(get_sim_func(name) if isinstance(name, str) else name) : params \
                   for name, params in simulations.items()}

    components = [func(n_seconds, fs, **params) for func, params in simulations.items()]

    proportions = repeat(proportions) if isinstance(proportions, int) else proportions

    sig = proportional_sum(components, proportions)

    return sig
