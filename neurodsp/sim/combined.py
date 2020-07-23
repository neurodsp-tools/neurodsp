"""Simulating time series, with combinations of periodic, aperiodic and transient components."""

from itertools import repeat

import numpy as np

from neurodsp.sim.info import get_sim_func
from neurodsp.utils.decorators import normalize

###################################################################################################
###################################################################################################

@normalize
def sim_combined(n_seconds, fs, components, component_variances=1):
    """Simulate a signal by combining multiple component signals.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    components : dictionary
        A dictionary of simulation functions to run, with their desired parameters.
    component_variances : list of float or 1, optional, default: 1
        Variance to simulate with for each component of the signal.
        If 1, each component signal is simulated with unit variance.

    Returns
    -------
    sig : 1d array
        Simulated combined signal.

    Examples
    --------
    Simulate a combined signal with an aperiodic and periodic component:

    >>> sim_components = {'sim_powerlaw': {'exponent' : -2}, 'sim_oscillation': {'freq' : 10}}
    >>> sig = sim_combined(n_seconds=1, fs=500, components=sim_components)

    Simulate a combined signal with multiple periodic components:

    >>> sim_components = {'sim_powerlaw': {'exponent' : -2},
    ...                   'sim_oscillation': [{'freq' : 10}, {'freq' : 20}]}
    >>> sig = sim_combined(n_seconds=1, fs=500, components=sim_components)

    Simulate a combined signal with unequal variance for the different components:

    >>> sig = sim_combined(n_seconds=1, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation': {'freq' : 10}},
    ...                    component_variances=[0.25, 0.75])
    """

    # Check how simulation components are specified, in terms of number of parameter sets
    n_sims = sum([1 if isinstance(params, dict) else len(params) \
        for params in components.values()])

    # Check that the variance definition matches the number of components specified
    if not (component_variances == 1 or len(component_variances) == n_sims):
        raise ValueError('Signal components and variances lengths do not match.')

    # Collect the sim function to use, and repeat variance if is single number
    components = {(get_sim_func(name) if isinstance(name, str) else name) : params \
                   for name, params in components.items()}
    variances = repeat(component_variances) if isinstance(component_variances, (int, float)) \
        else iter(component_variances)

    # Simulate each component of the signal
    sig_components = []
    for func, params in components.items():

        # If list, params should be a list of separate parameters for each function call
        if isinstance(params, list):
            sig_components.extend([func(n_seconds, fs, **cur_params, variance=next(variances)) \
                for cur_params in params])

        # Otherwise, params should be a dictionary of parameters for single call
        else:
            sig_components.append(func(n_seconds, fs, **params, variance=next(variances)))

    # Combine total signal across all simulated components
    sig = np.sum(sig_components, axis=0)

    return sig
