"""Simulation functions that return multiple instances."""

import numpy as np

from neurodsp.sim.signals import Simulations, VariableSimulations, MultiSimulations
from neurodsp.sim.generators import sig_yielder, sig_sampler
from neurodsp.sim.update import ParamIter
from neurodsp.utils.data import compute_nsamples

###################################################################################################
###################################################################################################

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

    sigs = np.zeros([n_sims, compute_nsamples(sim_params['n_seconds'], sim_params['fs'])])
    for ind, sig in enumerate(sig_yielder(sim_func, sim_params, n_sims)):
        sigs[ind, :] = sig

    if return_type == 'object':
        return Simulations(sigs, sim_params, sim_func)
    else:
        return sigs


def sim_across_values(sim_func, sim_params, n_sims, return_type='object'):
    """Simulate multiple signals across different parameter values.

    Parameters
    ----------
    sim_func : callable
        Function to create the simulated time series.
    sim_params : ParamIter or iterable or list of dict
        Simulation parameters for `sim_func`.
    n_sims : int
        Number of simulations to create per parameter definition.
    return_type : {'object', 'array'}
        Specifies the return type of the simulations.
        If 'object', returns simulations and metadata in a 'MultiSimulations' object.
        If 'array', returns the simulations (no metadata) in an array.

    Returns
    -------
    sims : MultiSimulations or array
        Simulations, return type depends on `return_type` argument.
        If array, signals are collected together as [n_sets, n_sims, sig_length].

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

    sims = MultiSimulations(update=getattr(sim_params, 'update', None),
                            component=getattr(sim_params, 'component', None))
    for ind, cur_sim_params in enumerate(sim_params):
        sims.add_signals(sim_multiple(sim_func, cur_sim_params, n_sims, 'object'))

    if return_type == 'array':
        sims = np.squeeze(np.array([el.signals for el in sims]))

    return sims


def sim_from_sampler(sim_func, sim_sampler, n_sims, return_type='object'):
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
        Specifies the return type of the simulations.
        If 'object', returns simulations and metadata in a 'VariableSimulations' object.
        If 'array', returns the simulations (no metadata) in an array.

    Returns
    -------
    sigs : VariableSimulations or 2d array
        Simulations, return type depends on `return_type` argument.
        If array, simulations are organized as [n_sims, sig length].

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
    n_samples = compute_nsamples(sim_sampler.params['n_seconds'], sim_sampler.params['fs'])
    sigs = np.zeros([n_sims, n_samples])
    for ind, (sig, params) in enumerate(sig_sampler(sim_func, sim_sampler, True, n_sims)):
        sigs[ind, :] = sig
        all_params[ind] = params

    if return_type == 'object':
        return VariableSimulations(sigs, all_params, sim_func)
    else:
        return sigs
