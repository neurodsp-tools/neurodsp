"""Simulation parameter related objects.

Notes:
- once we hit minimum python of 3.9, dictionary merging can by updated to use |
"""

from copy import deepcopy

###################################################################################################
###################################################################################################

BASE_PARAMS = ['n_seconds', 'fs']

## SIMULATION PARAMETER FUNCTIONS

def get_base_params(params):
    """Get base parameters from a parameter definition.

    Parameters
    ----------
    params : dict or list of dict or Object
        Parameter definition.

    Returns
    -------
    base : dict
        Base parameters.
    """

    from neurodsp.sim.update import BaseUpdater

    if isinstance(params, dict):
        base = _get_base_params(params)
    elif isinstance(params, list):
        base = _get_base_params(params[0])
    elif isinstance(params, BaseUpdater):
        base = getattr(params, 'base')
    else:
        raise ValueError('Parameter definition not understood.')

    return base


def _get_base_params(params):
    """Sub-function to get base parameters from a dictionary of parameters."""

    return {key : value for key, value in params.items() if key in BASE_PARAMS}


def drop_base_params(params):
    """Drop base parameters from a parameter definition.

    Parameters
    ----------
    params : dict or list of dict
        Parameter definition(s).

    Returns
    -------
    params : dict or list of dict
        Parameter definition(s), excluding base parameters.
    """

    if isinstance(params, dict):
        params = _drop_base_params(params)
    elif isinstance(params, list):
        params = [_drop_base_params(cparams) for cparams in params]
    else:
        raise ValueError('Parameter definition not understood.')

    return params


def _drop_base_params(params):
    """Sub-function to drop base parameters from a dictionary of parameters."""

    return {key : value for key, value in params.items() if key not in BASE_PARAMS}


def get_param_values(params, extract=None, component=None):
    """Get a set of parameter values from a set of parameter definitions.

    Parameters
    ----------
    params : list of dict
        Parameter definitions for multiple simulations.
    extract : str
        Name of the parameter to extract.
    component : str, optional
        Which component to extract the parameter from.
        Only used if the parameter definition is for a multi-component simulation.

    Returns
    -------
    values : list
        Extracted parameter values.
    """

    if component:
        values =[cparams['components'][component][extract] for cparams in params]
    elif extract:
        values = [cparams[extract] for cparams in params]
    else:
        values = None

    return values


## SIMULATION PARAMETER OBJECTS

class SimParams():
    """Object for managing simulation parameters.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.

    Attributes
    ----------
    base : dict
        Dictionary of base parameters, common across all parameter definitions.
    params : dict
        Dictionary of created simulation parameter definitions.
    """

    def __init__(self, n_seconds=None, fs=None):
        """Initialize SimParams object."""

        self.n_seconds = n_seconds
        self.fs = fs

        self._params = {}


    def __getitem__(self, label):
        """Make object subscriptable, to access stored simulation parameters.

        Parameters
        ----------
        label : str
            Label to access simulation parameters from `params`.
        """

        return {**self.base, **self._params[label]}


    def __contains__(self, label):
        """Define object contents as whether label exists in object.

        Parameters
        ----------
        label : str
            Label to check whether it exists in object.
        """

        return label in self.labels


    @property
    def base(self):
        """Get the base parameters, common across all simulations.

        Returns
        -------
        base_params : dict
            Definition of the current base parameters.
        """

        return {'n_seconds' : self.n_seconds, 'fs' : self.fs}


    @property
    def labels(self):
        """Get the set of labels for the defined parameters.

        Returns
        -------
        labels : list of str
            Labels for all defined simulation parameters.
        """

        return list(self._params.keys())


    @property
    def params(self):
        """Get the set of currently defined simulation parameters.

        Returns
        -------
        params : dict
            Dictionary of currently defined simulation parameters.
            Each key is a label for the parameter definition.
            Each value is a dictionary of simulations parameters.
        """

        return {label : {**self.base, **params} for label, params in self._params.items()}


    def make_params(self, parameters=None, **kwargs):
        """Make a simulation parameter definition from given parameters.

        Parameters
        ----------
        parameters : dict or list of dict
            Parameter definition(s) to create simulation parameter definition.
        **kwargs
            Additional keyword arguments to create the simulation definition.

        Returns
        -------
        params : dict
            Parameter definition.
        """

        return {**self.base, **self._make_params(parameters, **kwargs)}


    def register(self, label, parameters=None, **kwargs):
        """Register a new simulation parameter definition.

        Parameters
        ----------
        label : str
            Label to set simulation parameters under in `params`.
        parameters : dict or list of dict
            Parameter definition(s) to create simulation parameter definition.
        **kwargs
            Additional keyword arguments to create the simulation definition.
        """

        self._params[label] = self._make_params(parameters, **kwargs)


    def register_group(self, group, clear=False):
        """Register multiple simulation parameter definitions.

        Parameters
        ----------
        group : dict
            Dictionary of simulation parameters.
            Each key should be a label, and each set of values a dictionary of parameters.
        clear : bool, optional, default: False
            If True, clears current parameter definitions before adding new group.
        """

        if clear:
            self.clear()

        for label, parameters in group.items():
            self.register(label, parameters)


    def update_base(self, n_seconds=False, fs=False):
        """Update base parameters.

        Parameters
        ----------
        n_seconds : float, optional
            Simulation time, in seconds.
        fs : float, optional
            Sampling rate of simulated signal, in Hz.

        Notes
        -----
        If set as False, value will not be updated.
        If set as None, value will be updated to None.
        """

        if n_seconds is not False:
            self.n_seconds = n_seconds
        if fs is not False:
            self.fs = fs


    def update_param(self, label, attribute, new_value):
        """Update a simulation parameter definition.

        Parameters
        ----------
        label : str
            Label for the parameter definition.
        attribute : str
            Name of the attribute to update.
        new_value : obj
            New value to give the attribute.
        """

        self._params[label][attribute] = new_value


    def clear(self, clear_base=False):
        """"Clear parameter definitions.

        Parameters
        ----------
        clear_base : bool, optional, default: False
            Whether to also clear base parameters.
        """

        self._params = {}

        if clear_base:
            self.update_base(None, None)


    def to_iters(self):
        """Convert to a SimIters object.

        Returns
        -------
        iters : SimIters
            A converted object, initialized with the current base parameters.
        """

        iters = SimIters(**self.base)
        iters.register_group(self.params)

        return iters


    def to_samplers(self, n_samples=None):
        """Convert to a SimSamplers object.

        Parameters
        ----------
        n_samples : int, optional
            The number of parameter iterations to set as max.
            If None, samplers are created as infinite generators.

        Returns
        -------
        samplers : SimSamplers
            A converted object, initialized with the current base parameters.
        """

        samplers = SimSamplers(**self.base, n_samples=n_samples)
        samplers.register_group(self.params)

        return samplers


    def copy(self):
        """Return a copy of the current object."""

        return deepcopy(self)


    def _make_params(self, parameters=None, **kwargs):
        """Sub-function for `make_params`."""

        parameters = {} if not parameters else deepcopy(parameters)

        if isinstance(parameters, list):
            comps = [parameters.pop(0)]
            kwargs = {**kwargs, **parameters[0]} if parameters else kwargs
            params = self._make_combined_params(comps, **kwargs)
        else:
            params = {**parameters, **kwargs}

        # If any base parameters were passed in, clear them
        for bparam in self.base:
            params.pop(bparam, None)

        return params


    def _make_combined_params(self, components, component_variances=None):
        """Make parameters for combined simulations, specifying multiple components.

        Parameters
        ----------
        components : list of dict
            List of simulation component parameters.
        component_variances : list of float
            Component variances for the combined simulation.

        Returns
        -------
        params : dict
            Parameter definition.
        """

        parameters = {}

        comps = {}
        for comp in components:
            comps.update(**deepcopy(comp))
        parameters['components'] = comps

        if component_variances:
            parameters['component_variances'] = component_variances

        return parameters


class SimIters(SimParams):
    """Object for managing simulation iterators.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    """

    def __init__(self, n_seconds=None, fs=None):
        """Initialize SimIters objects."""

        SimParams.__init__(self, n_seconds, fs)

        self._iters = {}


    def __getitem__(self, label):
        """Make object subscriptable, to access simulation iterators.

        Parameters
        ----------
        label : str
            Label to access and create parameter iterator from `_iters`.
        """

        return self.make_iter(**self._iters[label])


    @property
    def iters(self):
        """Get the set of currently defined simulation iterators.

        Returns
        -------
        iters : dict
            Dictionary of currently defined simulation iterators.
            Each key is a label for the parameter iterator.
            Each value is a ParamIter object.
        """

        return {label : self.make_iter(**params) for label, params in self._iters.items()}


    @property
    def labels(self):
        """Get the set of labels for the defined iterators.

        Returns
        -------
        labels : list of str
            Labels for all defined iterators.
        """

        return list(self._iters.keys())


    def make_iter(self, label, update, values, component=None):
        """Create iterator to step across simulation parameter values.

        Parameters
        ----------
        label : str
            Label for the simulation parameters.
        update : str
            Name of the parameter to update.
        values : 1d array
            Values to iterate across.
        component : str, optional
            Which component to update the parameter in.
            Only used if the parameter definition is for a multi-component simulation.

        Returns
        -------
        ParamIter
            Generator object for iterating across simulation parameters.
        """

        from neurodsp.sim.update import ParamIter

        assert label in self._params.keys(), "Label for simulation parameters not found."

        return ParamIter(super().__getitem__(label), update, values, component)


    def register_iter(self, name, label, update, values, component=None):
        """Register an iterator definition.

        Parameters
        ----------
        name : str
            Name to give the registered iterator.
        label : str
            Label for the simulation parameters.
        update : str
            Name of the parameter to update.
        values : 1d array
            Values to iterate across.
        component : str, optional
            Which component to update the parameter in.
            Only used if the parameter definition is for a multi-component simulation.
        """

        self._iters[name] = {
            'label' : label,
            'update' : update,
            'values' : values,
            'component' : component,
        }


    def register_group_iters(self, group, clear=False):
        """Register a group of simulation iterators.

        Parameters
        ----------
        group : list of list or list of dict
            Set of simulation iterator definitions.
        clear : bool, optional, default: False
            If True, clears current parameter iterators before adding new group.
        """

        if clear:
            self.clear()

        for iterdef in group:
            if isinstance(iterdef, list):
                self.register_iter(*iterdef)
            elif isinstance(iterdef, dict):
                self.register_iter(**iterdef)


    def update_iter(self, label, attribute, new_value):
        """Update the definition of an iterator.

        Parameters
        ----------
        label : str
            Label for the iterator.
        attribute : str
            Name of the attribute to update.
        new_value : obj
            New value to give the attribute.
        """

        self._iters[label][attribute] = new_value


    def clear(self, clear_iters=True, clear_params=False, clear_base=False):
        """"Clear iterator and/or parameter definitions.

        Parameters
        ----------
        clear_iters : bool, optional, default: True
            Whether to clear the currently defined iterators.
        clear_params : bool, optional, default: False
            Whether to clear the currently defined simulation parameters.
        clear_base : bool, optional, default: False
            Whether to also clear base parameters.
            Only applied if `clear_params` is True.
        """

        if clear_iters:
            self._iters = {}

        if clear_params:
            super().clear(clear_base=clear_base)


class SimSamplers(SimParams):
    """Object for sampling simulation parameter definitions.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    n_samples : int, optional
        The number of parameter iterations to set as max.
        If None, samplers are created as infinite generators.
    """

    def __init__(self, n_seconds=None, fs=None, n_samples=None):
        """Initialize SimSamplers objects."""

        SimParams.__init__(self, n_seconds, fs)

        self.n_samples = n_samples
        self._samplers = {}


    def __getitem__(self, label):
        """Make object subscriptable, to access simulation iterators.

        Parameters
        ----------
        label : str
            Label to access and create parameter sampler from `_samplers`.
        """

        return self.make_sampler(**self._samplers[label])


    @property
    def labels(self):
        """Get the set of labels for the defined iterators."""

        return list(self._samplers.keys())


    @property
    def samplers(self):
        """Get the set of currently defined simulation samplers.

        Returns
        -------
        samplers : dict
            Dictionary of currently defined simulation samplers.
            Each key is a label for the parameter sampler.
            Each value is a ParamSampler object.
        """

        return {label : self.make_sampler(**params) for label, params in self._samplers.items()}


    def make_sampler(self, label, samplers, n_samples=None):
        """Create sampler to sample simulation parameter values.

        Parameters
        ----------
        label : str
            Label for the simulation parameters.
        samplers : dict
            Sampler definitions to update parameters with.
            Each key should be a callable, a parameter updater function.
            Each value should be a generator, to sample updated parameter values from.

        Returns
        -------
        ParamSampler
            Generator object for sampling simulation parameters.
        """

        from neurodsp.sim.update import ParamSampler

        return ParamSampler(super().__getitem__(label), samplers,
                            n_samples if n_samples else self.n_samples)


    def register_sampler(self, name, label, samplers):
        """Register a sampler definition.

        Parameters
        ----------
        name : str
            Name to give the registered iterator.
        label : str
            Label for the simulation parameters.
        samplers : dict
            Sampler definitions to update parameters with.
            Each key should be a callable, a parameter updater function.
            Each value should be a generator, to sample updated parameter values from.
        """

        self._samplers[name] = {
            'label' : label,
            'samplers' : samplers,
        }


    def register_group_samplers(self, group, clear=False):
        """Register a group of simulation samplers.

        Parameters
        ----------
        group : list of list or list of dict
            Set of simulation iterator definitions.
        clear : bool, optional, default: False
            If True, clears current parameter samplers before adding new group.
        """

        if clear:
            self.clear()

        for samplerdef in group:
            if isinstance(samplerdef, list):
                self.register_sampler(*samplerdef)
            elif isinstance(samplerdef, dict):
                self.register_sampler(**samplerdef)


    def update_sampler(self, label, attribute, new_value):
        """Update the definition of a sampler.

        Parameters
        ----------
        label : str
            Label for the sampler.
        attribute : str
            Name of the attribute to update.
        new_value : obj
            New value to give the attribute.
        """

        self._samplers[label][attribute] = new_value


    def clear(self, clear_samplers=True, clear_params=False, clear_base=False):
        """"Clear sampler and/or parameter definitions.

        Parameters
        ----------
        clear_samplers : bool, optional, default: True
            Whether to clear the currently defined samplers.
        clear_params : bool, optional, default: False
            Whether to clear the currently defined simulation parameters.
        clear_base : bool, optional, default: False
            Whether to also clear base parameters.
            Only applied if `clear_params` is True.
        """

        if clear_samplers:
            self._samplers = {}

        if clear_params:
            super().clear(clear_base=clear_base)
