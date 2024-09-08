"""Simulation functions that return multiple instances."""

from neurodsp.sim.signals import Simulations, VariableSimulations, MultiSimulations
from neurodsp.sim.generators import sig_yielder, sig_sampler
from neurodsp.sim.params import get_base_params
from neurodsp.sim.info import get_sim_func

###################################################################################################
###################################################################################################

def sim_multiple(sim_func, params, n_sims):
    """Simulate multiple samples of a specified simulation.

    Parameters
    ----------
    sim_func : callable
        Function to create the simulated time series.
        If string, should be the name of the desired simulation function.
    params : dict
        The parameters for the simulated signal, passed into `sim_func`.
    n_sims : int
        Number of simulations to create.

    Returns
    -------
    sims : Simulations
        Simulations object with simulated time series and metadata.
        Simulated signals are in the 'signals' attribute with shape [n_sims, sig_length].

    Examples
    --------
    Simulate multiple samples of a powerlaw signal:

    >>> from neurodsp.sim.aperiodic import sim_powerlaw
    >>> params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}
    >>> sims = sim_multiple(sim_powerlaw, params, n_sims=3)
    """

    sims = Simulations(n_sims, params, sim_func)
    for ind, sig in enumerate(sig_yielder(sim_func, params, n_sims)):
        sims.add_signal(sig, index=ind)

    return sims


def sim_across_values(sim_func, params):
    """Simulate signals across different parameter values.

    Parameters
    ----------
    sim_func : callable
        Function to create the simulated time series.
        If string, should be the name of the desired simulation function.
    params : ParamIter or iterable or list of dict
        Simulation parameters for `sim_func`.

    Returns
    -------
    sims : VariableSimulations
        Simulations object with simulated time series and metadata.
        Simulated signals are in the 'signals' attribute with shape [n_sims, sig_length].

    Examples
    --------
    Simulate multiple powerlaw signals using a ParamIter object:

    >>> from neurodsp.sim.aperiodic import sim_powerlaw
    >>> from neurodsp.sim.update import ParamIter
    >>> base_params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : None}
    >>> param_iter = ParamIter(base_params, 'exponent', [-2, 1, 0])
    >>> sims = sim_across_values(sim_powerlaw, param_iter)

    Simulate multiple powerlaw signals from manually defined set of simulation parameters:

    >>> params = [{'n_seconds' : 2, 'fs' : 250, 'exponent' : -2},
    ...           {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}]
    >>> sims = sim_across_values(sim_powerlaw, params)
    """

    sims = VariableSimulations(len(params), get_base_params(params), sim_func,
                               update=getattr(params, 'update', None),
                               component=getattr(params, 'component', None))

    sim_func = get_sim_func(sim_func)

    for ind, cur_params in enumerate(params):
        sims.add_signal(sim_func(**cur_params), cur_params, index=ind)

    return sims


def sim_multi_across_values(sim_func, params, n_sims):
    """Simulate multiple signals across different parameter values.

    Parameters
    ----------
    sim_func : callable
        Function to create the simulated time series.
        If string, should be the name of the desired simulation function.
    params : ParamIter or iterable or list of dict
        Simulation parameters for `sim_func`.
    n_sims : int
        Number of simulations to create per parameter definition.

    Returns
    -------
    sims : MultiSimulations
        Simulations object with simulated time series and metadata.
        Simulated signals are in the 'signals' attribute with shape [n_sets, n_sims, sig_length].

    Examples
    --------
    Simulate multiple powerlaw signals using a ParamIter object:

    >>> from neurodsp.sim.aperiodic import sim_powerlaw
    >>> from neurodsp.sim.update import ParamIter
    >>> base_params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : None}
    >>> param_iter = ParamIter(base_params, 'exponent', [-2, 1, 0])
    >>> sims = sim_multi_across_values(sim_powerlaw, param_iter, n_sims=2)

    Simulate multiple powerlaw signals from manually defined set of simulation parameters:

    >>> params = [{'n_seconds' : 2, 'fs' : 250, 'exponent' : -2},
    ...           {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}]
    >>> sims = sim_multi_across_values(sim_powerlaw, params, n_sims=2)
    """

    sims = MultiSimulations(update=getattr(params, 'update', None),
                            component=getattr(params, 'component', None))
    for cur_params in params:
        sims.add_signals(sim_multiple(sim_func, cur_params, n_sims))

    return sims


def sim_from_sampler(sim_func, sampler, n_sims):
    """Simulate a set of signals from a parameter sampler.

    Parameters
    ----------
    sim_func : str callable
        Function to create the simulated time series.
        If string, should be the name of the desired simulation function.
    sampler : ParamSampler
        Parameter definition to sample from.
    n_sims : int
        Number of simulations to create per parameter definition.

    Returns
    -------
    sims : VariableSimulations
        Simulations object with simulated time series and metadata.
        Simulated signals are in the 'signals' attribute with shape [n_sims, sig_length].

    Examples
    --------
    Simulate multiple powerlaw signals using a parameter sampler:

    >>> from neurodsp.sim.aperiodic import sim_powerlaw
    >>> from neurodsp.sim.update import create_updater, create_sampler, ParamSampler
    >>> params = {'n_seconds' : 10, 'fs' : 250, 'exponent' : None}
    >>> samplers = {create_updater('exponent') : create_sampler([-2, -1, 0])}
    >>> param_sampler = ParamSampler(params, samplers)
    >>> sims = sim_from_sampler(sim_powerlaw, param_sampler, n_sims=2)
    """

    sims = VariableSimulations(n_sims, get_base_params(sampler), sim_func)
    for ind, (sig, params) in enumerate(sig_sampler(sim_func, sampler, True, n_sims)):
        sims.add_signal(sig, params, index=ind)

    return sims
