"""Simulation functions that return multiple instances."""

import numpy as np

from neurodsp.utils.core import counter

###################################################################################################
###################################################################################################

def sig_yielder(sim_func, sim_params, n_sims):
    """Generator to yield simulated signals from a given simulation function and parameters.

    Parameters
    ----------
    sim_func : callable
        Function to create the simulated time series.
    sim_params : dict
        The parameters for the simulated signal, passed into `sim_func`.
    n_sims : int, optional
        Number of simulations to set as the max.
        If None, creates an infinite generator.

    Yields
    ------
    sig : 1d array
        Simulated time series.
    """

    for _ in counter(n_sims):
        yield sim_func(**sim_params)


def sig_sampler(sim_func, sim_params, return_sim_params=False, n_sims=None):
    """Generator to yield simulated signals from a parameter sampler.

    Parameters
    ----------
    sim_func : callable
        Function to create the simulated time series.
    sim_params : iterable
        The parameters for the simulated signal, passed into `sim_func`.
    return_sim_params : bool, optional, default: False
        Whether to yield the simulation parameters as well as the simulated time series.
    n_sims : int, optional
        Number of simulations to set as the max.
        If None, creates an infinite generator.

    Yields
    ------
    sig : 1d array
        Simulated time series.
    sample_params : dict
        Simulation parameters for the yielded time series.
    """

    if len(sim_params) and n_sims and n_sims > len(sim_params):
        msg = 'Cannot simulate the requested number of sims with the given parameters.'
        raise ValueError(msg)

    for ind, sample_params in zip(counter(n_sims), sim_params):

        if return_sim_params:
            yield sim_func(**sample_params), sample_params
        else:
            yield sim_func(**sample_params)

        if n_sims and ind >= n_sims:
            break


def sim_multiple(sim_func, sim_params, n_sims):
    """Simulate multiple samples of a specified simulation.

    Parameters
    ----------
    sim_func : callable
        Function to create the simulated time series.
    sim_params : dict
        The parameters for the simulated signal, passed into `sim_func`.
    n_sims : int
        Number of simulations to create.

    Returns
    -------
    sigs : 2d array
        Simulations, as [n_sims, sig length].
    """

    sigs = np.zeros([n_sims, sim_params['n_seconds'] * sim_params['fs']])
    for ind, sig in enumerate(sig_yielder(sim_func, sim_params, n_sims)):
        sigs[ind, :] = sig

    return sigs


def sim_across_values(sim_func, sim_params, n_sims, output='dict'):
    """Simulate multiple signals across different parameter values.

    Parameters
    ----------
    sim_func : callable
        Function to create the simulated time series.
    sim_params : iterable or list of dict
        Simulation parameters for `sim_func`.
    n_sims : int
        Number of simulations to create per parameter definition.
    output : {'dict', 'array'}
        Organization of the output for the sims.
        If 'dict', stored in a dictionary, organized by simulation parameter.
        If 'array', all sims are organized into a 2D array.

    Returns
    -------
    sims : dict of {float : array} or array
        If dict, dictionary of simulated signals, where:
            Each key is the simulation parameter value for the set of simulations.
            Each value is the set of simulations for that value, as [n_sims, sig_length].
        If array, is all signals collected together as [n_sims, sig_length].
    """

    sims = {}
    for ind, cur_sim_params in enumerate(sim_params):
        label = sim_params.values[ind] if hasattr(sim_params, 'values') else ind
        label = label[-1] if isinstance(label, list) else label
        sims[label] = sim_multiple(sim_func, cur_sim_params, n_sims)
    if output == 'array':
        sims = np.squeeze(np.array(list(sims.values())))

    return sims


def sim_from_sampler(sim_func, sim_sampler, n_sims, return_params=False):
    """Simulate a set of signals from a parameter sampler.

    Parameters
    ----------
    sim_func : callable
        Function to create the simulated time series.
    sim_sampler : ParamSampler
        Parameter definition to sample from.
    n_sims : int
        Number of simulations to create per parameter definition.
    return_params : bool, default: False
        Whether to collect and return the parameters of all the generated simulations.

    Returns
    -------
    sigs : 2d array
        Simulations, as [n_sims, sig length].
    all_params : list of dict
        Simulation parameters for each returned time series.
        Only returned if `return_params` is True.
    """

    all_params = [None] * n_sims
    sigs = np.zeros([n_sims, sim_sampler.params['n_seconds'] * sim_sampler.params['fs']])
    for ind, (sig, params) in enumerate(sig_sampler(sim_func, sim_sampler, True, n_sims)):
        sigs[ind, :] = sig

        if return_params:
            all_params[ind] = params

    if return_params:
        return sigs, all_params
    else:
        return sigs
