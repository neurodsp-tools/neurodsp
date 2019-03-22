"""Simulating time series, with combinations of periodic, aperiodic and transient components."""

from itertools import repeat

from neurodsp.sim.info import get_sim_func
from neurodsp.sim.utils import normalized_sum, proportional_sum

###################################################################################################
###################################################################################################

def sim_combined(n_seconds, fs, freq,
                 sim_periodic='oscillation', sim_aperiodic='powerlaw',
                 periodic_args={}, aperiodc_args={},
                 ratio=1., select_nonzero=True):
    """Simulate a combined signal, with periodic and aperiodic components.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    sim_periodic : str or executable
        Periodic simulation function to use.
    sim_aperiodic : str or executable
        Aperiodic simulation function to use.
    periodic_args : dictionary
        Parameters for the periodic sim function.
    aperiodic_args : dictionary
        Parameters for the aperiodic sim function.
    ratio : float, optional, default: 1.
        Ratio of signal variance. If >1, periodic is stronger, if <1 - aperiodic is stronger.
    select_nonzero : bool, optional, default: True
        Whether to select only non-zero samples when normalizing variance of the periodic signal.

    Returns
    -------
    sig : 1d array
        Simulated combined signal.
    """

    sim_pe = get_sim_func(sim_periodic) if isinstance(sim_periodic, str) else sim_periodic
    sim_ap = get_sim_func(sim_aperiodic) if isinstance(sim_aperiodic, str) else sim_aperiodic

    sig = normalized_sum(sim_pe(n_seconds, fs, freq, **periodic_args),
                       sim_ap(n_seconds, fs, **aperiodc_args),
                       ratio, select_nonzero)

    return sig


def sim_multiple_combined(n_seconds, fs, sim_funcs, sim_params, proportions=1.):
    """Sim multiple component signals and combine them.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    sim_funcs : list of str or executable
        Simulate functions for each component signal.
    sim_params : list of dictionaries
        Input parameters for each component signal.
    proportions : float of list of float
        Variance proportions to use to sum across signals.

    Returns
    -------
    sig : 1d array
        Simulated combined signal.
    """

    assert len(sim_funcs) == len(sim_params)

    sim_funcs = [get_sim_func(sim_func) for sim_func in sim_funcs]

    components = [func(n_seconds, fs, **params) for func, params in zip(sim_funcs, sim_params)]

    proportions = repeat(proportions) if isinstance(proportions, int) else proportions
    sig = proportional_sum(components, proportions)

    return sig
