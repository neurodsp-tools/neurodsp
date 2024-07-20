"""Simulation parameter management and updaters."""

from copy import deepcopy

import numpy as np

from neurodsp.sim.multi import sig_yielder
from neurodsp.utils.core import counter

###################################################################################################
###################################################################################################

## PARAM UPDATERS

def param_updater(parameter):
    """Create a lambda updater function to update a specified parameter.

    Parameters
    ----------
    parameter : str
        Name of the parameter to update.

    Returns
    -------
    callable
        Updater function which can update specified parameter in simulation parameters.
    """

    return lambda params, value : params.update({parameter : value})


def component_updater(parameter, component):
    """Create a lambda updater function to update a parameter within a simulation component.

    Parameters
    ----------
    parameter : str
        Name of the parameter to update.
    component : str
        Name of the component to update the parameter within.

    Returns
    -------
    callable
        Updater function which can update specified parameter in simulation parameters.
    """

    return lambda params, value : params['components'][component].update({parameter : value})


def create_updater(update, component=None):
    """Create an updater function for updating simulation parameters.

    Parameters
    ----------
    update : str
        Name of the parameter to update.
    component : str
        Name of the component to update the parameter within.

    Returns
    -------
    callable
        Updater function which can update specified parameter in simulation parameters.
    """

    if component is not None:
        updater = component_updater(update, component)
    else:
        updater = param_updater(update)

    return updater


## PARAM ITER

def param_iter_yielder(sim_params, updater, values):
    """Parameter yielder.

    Parameters
    ----------
    sim_params : dict
        Parameter definition.
    updater : callable
        Updater function to update parameter definition.
    values : 1d array
        Values to iterate across.

    Yields
    ------
    sim_params : dict
        Simulation parameter definition.
    """

    sim_params = deepcopy(sim_params)

    for value in values:
        updater(sim_params, value)
        yield deepcopy(sim_params)


class ParamIter():
    """Object for iterating across parameter updates.

    Parameters
    ----------
    params : dict
        Parameter definition to create iterator with.
    update : str
        Name of the parameter to update.
    values : 1d array
        Values to iterate across.
    component : str, optional
        Which component to update the parameter in.
        Only used if the parameter definition is for a multi-component simulation.

    Attributes
    ----------
    index : int
        Index of current location through the iteration.
    yielder : generator
        Generator for sampling the sig iterations.
    """

    def __init__(self, params, update, values, component=None):
        """Initialize parameter iteration object."""

        params = deepcopy(params)

        if component is not None:
            params['components'][component][update] = None
        else:
            params[update] = None

        self.params = params
        self.update = update
        self.values = values
        self.component = component

        self._updater = create_updater(self.update, self.component)

        self.index = 0
        self.yielder = None
        self._reset_yielder()


    def __next__(self):
        """Sample the next set of simulation parameters."""

        self.index += 1
        return next(self.yielder)


    def __iter__(self):
        """Iterate across simulation parameters."""

        self._reset_yielder()
        for _ in counter(len(self)):
            yield next(self)


    def __len__(self):
        """Define length of the object as the number of values to step across."""

        return len(self.values)


    def _reset_yielder(self):
        """Reset the object yielder."""

        self.index = 0
        self.yielder = param_iter_yielder(self.params, self._updater, self.values)


## PARAM SAMPLERS

def create_sampler(values, probs=None, n_samples=None):
    """Create a generator to sample from a parameter range.

    Parameters
    ----------
    values : list or 1d array
        Parameter values to create a generator for.
    probs : 1d array, optional
        Probabilities to sample from values.
        If provided, should be the same lengths as `values`.
    n_samples : int, optional
        The number of parameter iterations to set as max.
        If None, creates an infinite generator.

    Yields
    ------
    generator
        Generator to sample parameter values from.
    """

    # Check that length of values is same as length of probs, if provided
    if np.any(probs):
        if len(values) != len(probs):
            raise ValueError("The number of options must match the number of probabilities.")

    for _ in counter(n_samples):

        if isinstance(values[0], (list, np.ndarray)):
            yield values[np.random.choice(len(values), p=probs)]
        else:
            yield np.random.choice(values, p=probs)


def param_sample_yielder(sim_params, samplers, n_samples=None):
    """Generator to yield randomly sampled parameter definitions.

    Parameters
    ----------
    sim_params : dict
        The parameters for the simulated signal.
    samplers : dict
        Sampler definitions to update parameters with.
        Each key should be a callable, a parameter updated function.
        Each value should be a generator, to sample updated parameter values from.
    n_samples : int, optional
        The number of parameter iterations to set as max.
        If None, creates an infinite generator.

    Yields
    ------
    sim_params : dict
        Simulation parameter definition.
    """

    for ind in counter(n_samples):
        out_params = deepcopy(sim_params)
        for updater, sampler in samplers.items():
            updater(out_params, next(sampler))

        yield out_params


class ParamSampler():
    """Object for sampling parameter definitions.

    Parameters
    ----------
    params : dict
        Parameter definition to create sampler with.
    samplers : dict
        Sampler definitions to update parameters with.
        Each key should be a callable, a parameter updated function.
        Each value should be a generator, to sample updated parameter values from.
    n_samples : int, optional
        The number of parameter iterations to set as max.
        If None, creates an infinite generator.

    Attributes
    ----------
    index : int
        Index of current number of yielded parameter definitions.
    yielder : generator
        Generator for sampling the parameter samples.
    """

    def __init__(self, params, samplers, n_samples=None):
        """Initialize parameter sampler object."""

        self.params = deepcopy(params)
        self.samplers = samplers
        self.n_samples = n_samples

        self.yielder = None
        self._reset_yielder()


    def __next__(self):
        """Sample the next set of simulation parameters."""

        return next(self.yielder)


    def __iter__(self):
        """Iterate across sampled simulation parameters."""

        self._reset_yielder()
        for _ in counter(len(self)):
            yield next(self)


    def __len__(self):
        """Define length of the object as the maximum number of parameters to sample."""

        return self.n_samples if self.n_samples else 0


    def _reset_yielder(self):
        """Reset the object yielder."""

        self.yielder = param_sample_yielder(self.params, self.samplers, self.n_samples)


## SIG ITER

class SigIter():
    """Object for iterating across sampled simulations.

    Parameters
    ----------
    sim_func : callable
        Function to create simulations.
    sim_params : dict
        Simulation parameters.
    n_sims : int, optional
        Number of simulations to create.
        If None, creates an infinite generator.

    Attributes
    ----------
    index : int
        Index of current location through the iteration.
    yielder : generator
        Generator for sampling the sig iterations.
    """

    def __init__(self, sim_func, sim_params, n_sims=None):
        """Initialize signal iteration object."""

        self.sim_func = sim_func
        self.sim_params = deepcopy(sim_params)
        self.n_sims = n_sims

        self.index = 0
        self.yielder = None
        self._reset_yielder()


    def __next__(self):
        """Sample a new simulation."""

        self.index += 1

        return next(self.yielder)


    def __iter__(self):
        """Iterate across simulation outputs."""

        self._reset_yielder()
        for _ in counter(len(self)):
            yield next(self)


    def __len__(self):
        """Define length of the object as the number of simulations to create."""

        return self.n_sims if self.n_sims else 0


    def _reset_yielder(self):
        """Reset the object yielder."""

        self.index = 0
        self.yielder = sig_yielder(self.sim_func, self.sim_params, self.n_sims)
