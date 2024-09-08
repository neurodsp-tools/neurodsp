"""Generator simulation functions."""

from collections.abc import Sized

from neurodsp.sim.info import get_sim_func
from neurodsp.utils.core import counter

###################################################################################################
###################################################################################################

def sig_yielder(sim_func, params, n_sims):
    """Generator to yield simulated signals from a given simulation function and parameters.

    Parameters
    ----------
    sim_func : str or callable
        Function to create the simulated time series.
        If string, should be the name of the desired simulation function.
    params : dict
        The parameters for the simulated signal, passed into `sim_func`.
    n_sims : int, optional
        Number of simulations to set as the max.
        If None, creates an infinite generator.

    Yields
    ------
    sig : 1d array
        Simulated time series.
    """

    sim_func = get_sim_func(sim_func)
    for _ in counter(n_sims):
        yield sim_func(**params)


def sig_sampler(sim_func, params, return_params=False, n_sims=None):
    """Generator to yield simulated signals from a parameter sampler.

    Parameters
    ----------
    sim_func : str or callable
        Function to create the simulated time series.
        If string, should be the name of the desired simulation function.
    params : iterable
        The parameters for the simulated signal, passed into `sim_func`.
    return_params : bool, optional, default: False
        Whether to yield the simulation parameters as well as the simulated time series.
    n_sims : int, optional
        Number of simulations to set as the max.
        If None, length is defined by the length of `params`, and could be infinite.

    Yields
    ------
    sig : 1d array
        Simulated time series.
    sample_params : dict
        Simulation parameters for the yielded time series.
        Only returned if `return_params` is True.
    """

    sim_func = get_sim_func(sim_func)

    # If `params` has a size, and `n_sims` is defined, check that they are compatible
    #   To do so, we first check if the iterable has a __len__ attr, and if so check values
    if isinstance(params, Sized) and len(params) and n_sims and n_sims > len(params):
        msg = 'Cannot simulate the requested number of sims with the given parameters.'
        raise ValueError(msg)

    for ind, sample_params in zip(counter(n_sims), params):

        if return_params:
            yield sim_func(**sample_params), sample_params
        else:
            yield sim_func(**sample_params)

        if n_sims and ind >= n_sims:
            break
