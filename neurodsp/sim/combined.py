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

    # Check how simulations are specified, in terms of number of parameter sets
    n_sims = sum([1 if isinstance(params, dict) else len(params) \
        for params in simulations.values()])

    # Check that the variance definition matches the number of components specified
    if not (variances == 1 or len(variances) == n_sims):
        raise ValueError('Simulations and proportions lengths do not match.')

    # Collect the sim function to use, and repeat variance if is set to 1
    simulations = {(get_sim_func(name) if isinstance(name, str) else name) : params \
                   for name, params in simulations.items()}
    variances = repeat(variances) if isinstance(variances, int) else variances

    # Simulate each component, specifying the variance for each
    components = []
    for (func, params), variance in zip(simulations.items(), variances):

        # If list, params should be a list of separate parameters for each fucntion call
        if isinstance(params, list):
            components.extend([func(n_seconds, fs, **cur_params, variance=variance) \
                for cur_params in params])

        # Otherwise, params should be a dictionary of parameters for single call
        else:
            components.append(func(n_seconds, fs, **params, variance=variance))

    # Combine total signal across all simulated components
    sig = np.sum(components, axis=0)

    return sig
