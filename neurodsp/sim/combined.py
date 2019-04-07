"""Simulating time series, with combinations of periodic, aperiodic and transient components."""

from itertools import repeat

import numpy as np

from neurodsp.utils.decorators import normalize
from neurodsp.sim.info import get_sim_func

###################################################################################################
###################################################################################################

@normalize
def sim_combined(n_seconds, fs, simulations, variances=1):
    """Sim multiple component signals and combine them.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    simulations : dictionary
        A dictionary of simulation functions to run, with their desired parameters.
    variances : list of float or 1
        Specified variance for each component of the signal.

    Returns
    -------
    sig : 1d array
        Simulated combined signal.
    """

    if not (variances == 1 or len(variances) == len(simulations)):
        raise ValueError('Simulations and proportions lengths do not match.')

    # Collect the sim function to use, and repeat variance if is set to 1
    simulations = {(get_sim_func(name) if isinstance(name, str) else name) : params \
                   for name, params in simulations.items()}
    variances = repeat(variances) if isinstance(variances, int) else variances

    # Simulate each component, specifying variance, and combine them
    components = [func(n_seconds, fs, **params, variance=variance) for \
        (func, params), variance in zip(simulations.items(), variances)]
    sig = np.sum(components, axis=0)

    return sig
