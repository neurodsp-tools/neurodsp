"""Simulation functions that return multiple instances."""

from collections.abc import Sized

import numpy as np

from neurodsp.utils.core import counter
from neurodsp.sim.sims import Simulations, SampledSimulations

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
        If None, length is defined by the length of `sim_params`, and could be infinite.

    Yields
    ------
    sig : 1d array
        Simulated time series.
    sample_params : dict
        Simulation parameters for the yielded time series.
        Only returned if `return_sim_params` is True.
    """

    # If `sim_params` has a size, and `n_sims` is defined, check that they are compatible
    #   To do so, we first check if the iterable has a __len__ attr, and if so check values
    if isinstance(sim_params, Sized) and len(sim_params) and n_sims and n_sims > len(sim_params):
        msg = 'Cannot simulate the requested number of sims with the given parameters.'
        raise ValueError(msg)

    for ind, sample_params in zip(counter(n_sims), sim_params):

        if return_sim_params:
            yield sim_func(**sample_params), sample_params
        else:
            yield sim_func(**sample_params)

        if n_sims and ind >= n_sims:
            break


def sim_multiple(sim_func, sim_params, n_sims, return_type='object'):
    """Simulate multiple samples of a specified simulation.

    Parameters
    ----------
    sim_func : callable
        Function to create the simulated time series.
    sim_params : dict
        The parameters for the simulated signal, passed into `sim_func`.
    n_sims : int
        Number of simulations to create.
    return_type : {'object', 'array'}
        Specifies the return type of the simulations.
        If 'object', returns simulations and metadata in a 'Simulations' object.
        If 'array', returns the simulations (no metadata) in an array.

    Returns
    -------
    sigs : Simulations or 2d array
        Simulations, return type depends on `return_type` argument.
        Simulated time series are organized as [n_sims, sig length].

    Examples
    --------
    Simulate multiple samples of a powerlaw signal:

    >>> from neurodsp.sim.aperiodic import sim_powerlaw
    >>> params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}
    >>> sigs = sim_multiple(sim_powerlaw, params, n_sims=3)
    """

    sigs = np.zeros([n_sims, sim_params['n_seconds'] * sim_params['fs']])
    for ind, sig in enumerate(sig_yielder(sim_func, sim_params, n_sims)):
        sigs[ind, :] = sig

    if return_type == 'object':
        return Simulations(sigs, sim_params, sim_func)
    else:
        return sigs


def sim_across_values(sim_func, sim_params, n_sims, output='dict'):
    """Simulate multiple signals across different parameter values.

    Parameters
    ----------
    sim_func : callable
        Function to create the simulated time series.
    sim_params : ParamIter or iterable or list of dict
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

    Examples
    --------
    Simulate multiple powerlaw signals using a ParamIter object:

    >>> from neurodsp.sim.aperiodic import sim_powerlaw
    >>> from neurodsp.sim.params import ParamIter
    >>> base_params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : None}
    >>> param_iter = ParamIter(base_params, 'exponent', [-2, 1, 0])
    >>> sigs = sim_across_values(sim_powerlaw, param_iter, n_sims=2)

    Simulate multiple powerlaw signals from manually defined set of simulation parameters:

    >>> params = [{'n_seconds' : 2, 'fs' : 250, 'exponent' : -2},
    ...           {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}]
    >>> sigs = sim_across_values(sim_powerlaw, params, n_sims=2)
    """

    sims = {}
    for ind, cur_sim_params in enumerate(sim_params):
        label = sim_params.values[ind] if hasattr(sim_params, 'values') else ind
        label = label[-1] if isinstance(label, list) else label
        sims[label] = sim_multiple(sim_func, cur_sim_params, n_sims, 'array')
    if output == 'array':
        sims = np.squeeze(np.array(list(sims.values())))

    return sims


def sim_from_sampler(sim_func, sim_sampler, n_sims, return_type='object', return_params=False):
    """Simulate a set of signals from a parameter sampler.

    Parameters
    ----------
    sim_func : callable
        Function to create the simulated time series.
    sim_sampler : ParamSampler
        Parameter definition to sample from.
    n_sims : int
        Number of simulations to create per parameter definition.

    return_type : {'object', 'array'}
        XX
    #return_params : bool, default: False
    #    Whether to collect and return the parameters of all the generated simulations.

    Returns
    -------
    sigs : 2d array
        Simulations, as [n_sims, sig length].
    all_params : list of dict
        Simulation parameters for each returned time series.
        Only returned if `return_params` is True.

    Examples
    --------
    Simulate multiple powerlaw signals using a parameter sampler:

    >>> from neurodsp.sim.aperiodic import sim_powerlaw
    >>> from neurodsp.sim.update import create_updater, create_sampler, ParamSampler
    >>> params = {'n_seconds' : 10, 'fs' : 250, 'exponent' : None}
    >>> samplers = {create_updater('exponent') : create_sampler([-2, -1, 0])}
    >>> param_sampler = ParamSampler(params, samplers)
    >>> sigs = sim_from_sampler(sim_powerlaw, param_sampler, n_sims=2)
    """

    all_params = [None] * n_sims
    sigs = np.zeros([n_sims, sim_sampler.params['n_seconds'] * sim_sampler.params['fs']])
    for ind, (sig, params) in enumerate(sig_sampler(sim_func, sim_sampler, True, n_sims)):
        sigs[ind, :] = sig
        all_params[ind] = params

    if return_type == 'object':
        return SampledSimulations(sigs, all_params, sim_func)
    else:
        return sigs
