"""Objects for managing groups of simulated signals."""

from itertools import repeat

import numpy as np

from neurodsp.utils.core import listify
from neurodsp.utils.data import compute_nsamples
from neurodsp.sim.params import get_base_params, drop_base_params, get_param_values

###################################################################################################
###################################################################################################

class Simulations():
    """Data object for a set of simulated signals.

    Parameters
    ----------
    signals : 1d or 2nd array or int, optional
        If array, the simulated signals, organized as [n_sims, sig_length].
        If int, the number of expected simulations, used to pre-initialize array.
    params : dict, optional
        The simulation parameters that were used to create the simulations.
    sim_func : str or callable, optional
        The simulation function that was used to create the simulations.
        If callable, the name of the function is taken to be added to the object.

    Notes
    -----
    This object stores a set of simulations generated from a shared parameter definition.
    """

    def __init__(self, signals=None, params=None, sim_func=None):
        """Initialize Simulations object."""

        if signals is None:
            signals = np.array([])
        elif isinstance(signals, int):
            n_samples = compute_nsamples(params['n_seconds'], params['fs'])
            signals = np.zeros((signals, n_samples))
        self.signals = np.atleast_2d(signals)

        self._base_params = None
        self._params = None
        self.add_params(params)

        self.sim_func = sim_func.__name__ if callable(sim_func) else sim_func


    def __iter__(self):
        """Define iteration as stepping across individual simulated signals."""

        for sig in self.signals:
            yield sig


    def __getitem__(self, ind):
        """Define indexing as accessing simulated signals."""

        return self.signals[ind, :]


    def __len__(self):
        """Define the length of the object as the number of signals."""

        return len(self.signals)


    @property
    def n_seconds(self):
        """Alias n_seconds as a property attribute from base parameters."""

        return self._base_params['n_seconds'] if self.has_params else None


    @property
    def fs(self):
        """Alias fs as a property attribute from base parameters."""

        return self._base_params['fs'] if self.has_params else None


    @property
    def params(self):
        """Define the full set of simulation parameters (base + additional parameters)."""

        if self.has_params:
            params = {**self._base_params, **self._params}
        else:
            params = None

        return params


    @property
    def has_params(self):
        """Indicator for if the object has parameters."""

        return bool(self._params)


    @property
    def has_signals(self):
        """Indicator for if the object has signals."""

        return bool(len(self))


    def add_params(self, params):
        """Add parameter definition to object.

        Parameters
        ----------
        params : dict or None
            The simulation parameter definition(s).
        """

        if params:
            self._base_params = get_base_params(params)
            self._params = drop_base_params(params)


    def add_signal(self, signal, index=None):
        """Add a signal to the current object.

        Parameters
        ----------
        signal : 1d array
            A simulated signal to add to the object.
        index : int
            Index to insert the new signal in the signals attribute.
        """

        if index is not None:
            self.signals[index, :] = signal
        else:
            if not self.signals.size:
                self.signals = np.atleast_2d(signal)
            else:
                try:
                    self.signals = np.vstack([self.signals, signal])
                except ValueError as array_value_error:
                    msg = 'Size of the added signal is not consistent with existing signals.'
                    raise ValueError(msg) from array_value_error


class VariableSimulations(Simulations):
    """Data object for a set of simulated signals with variable parameter definitions.

    Parameters
    ----------
    signals : 2nd array or int, optional
        If array, the simulated signals, organized as [n_sims, sig_length].
        If int, the number of expected simulations, used to pre-initialize array.
    params : list of dict, optional
        The simulation parameters for each of the simulations.
    sim_func : str, optional
        The simulation function that was used to create the simulations.
    update : str
        The name of the parameter that is updated across simulations.
    component : str
        Which component the updated parameter is part of.
        Only used if the parameter definition is for a multi-component simulation.

    Notes
    -----
    This object stores a set of simulations with different parameter definitions per signal.
    """

    def __init__(self, signals=None, params=None, sim_func=None, update=None, component=None):
        """Initialize SampledSimulations object."""

        Simulations.__init__(self, signals, params, sim_func)
        if isinstance(signals, int):
            self._params = [{}] * signals
        self.update = update
        self.component = component


    @property
    def n_seconds(self):
        """Alias n_seconds as a property."""

        return self.params[0].n_seconds if self.has_params else None


    @property
    def fs(self):
        """Alias fs as a property."""

        return self.params[0].fs if self.has_params else None


    @property
    def params(self):
        """Define simulation parameters (base + additional parameters) for each simulation."""

        if self.has_params:
            params = [{**self._base_params, **self._params[ind]} for ind in range(len(self))]
        else:
            params = None

        return params


    @property
    def values(self):
        """Alias in the parameter definition of the parameter that varies across the sets."""

        return get_param_values(self.params, self.update, self.component)


    def add_params(self, params, index=None):
        """Add parameter definition(s) to object.

        Parameters
        ----------
        params : dict or list of dict, optional
            The simulation parameter definition(s).
        index : int
            Index to insert the new parameter definition.
        """

        if params:

            params = listify(params)

            base_params = get_base_params(params[0])
            if not self._base_params:
                self._base_params = base_params
            else:
                msg = 'Base params have to match existing parameters.'
                assert base_params == self._base_params, msg

            cparams = [drop_base_params(el) for el in params]
            if cparams[0]:
                if not self.has_params:
                    if len(self) > 1 and len(self) > len(cparams):
                        msg = 'Cannot add parameters to object without existing parameter values.'
                        raise ValueError(msg)
                    self._params = cparams
                else:
                    if index is not None:
                        self._params[index] = cparams[0]
                    else:
                        self._params.extend(cparams)

        else:
            if self.has_params:
                raise ValueError('Must add parameters if object already has them.')


    def add_signal(self, signal, params=None, index=None):
        """Add a signal to the current object.

        Parameters
        ----------
        signal : 1d array
            A simulated signal to add to the object.
        params : dict, optional
            Parameter definition for the added signal.
            If current object does not include parameters, should be empty.
            If current object does include parameters, this input is required.
        index : int
            Index to insert the new signal in the signals attribute.
        """

        super().add_signal(signal, index=index)
        self.add_params(params, index=index)


class MultiSimulations():
    """Data object for multiple sets of simulated signals.

    Parameters
    ----------
    signals : list of 2d array
        Sets of simulated signals, with each array organized as [n_sims, sig_length].
    params : list of dict
        The simulation parameters that were used to create the simulations.
    sim_func : str or list of str
        The simulation function(s) that were used to create the simulations.
    update : str
        The name of the parameter that is updated across sets of simulations.
    component : str
        Which component the updated parameter is part of.
        Only used if the parameter definition is for a multi-component simulation.

    Notes
    -----
    This object stores a set of simulations with multiple instances per parameter definition.
    """

    def __init__(self, signals=None, params=None, sim_func=None, update=None, component=None):
        """Initialize MultiSimulations object."""

        self.signals = []
        self.add_signals(signals, params, sim_func)
        self.update = update
        self.component = component


    def __iter__(self):
        """Define iteration as stepping across sets of simulated signals."""

        for sigs in self.signals:
            yield sigs


    def __getitem__(self, index):
        """Define indexing as accessing sets of simulated signals."""

        return self.signals[index]


    def __len__(self):
        """Define the length of the object as the number of sets of signals."""

        return len(self.signals)


    @property
    def n_seconds(self):
        """Alias n_seconds as a property."""

        return self.signals[0].n_seconds if self else None


    @property
    def fs(self):
        """Alias fs as a property."""

        return self.signals[0].fs if self else None


    @property
    def sim_func(self):
        """Alias func as property."""

        return self.signals[0].sim_func if self else None


    @property
    def params(self):
        """Alias in the set of parameters across all sets of simulations."""

        params = [self[ind].params for ind in range(len(self))]

        return params


    @property
    def values(self):
        """Alias in the parameter definition of the parameter that varies across the sets."""

        return get_param_values(self.params, self.update, self.component)


    @property
    def _base_params(self):
        """Alias base parameters as property."""

        return self.signals[0]._base_params if self else None


    @property
    def has_signals(self):
        """Indicator for if the object has signals."""

        return bool(len(self))


    def add_signals(self, signals, params=None, sim_func=None):
        """Add a set of signals to the current object.

        Parameters
        ----------
        signals : 2d array or list of 2d array or Simulations
            A set of simulated signals, organized as [n_sims, sig_length].
        params : dict or list of dict, optional
            The simulation parameters that were used to create the set of simulations.
        sim_func : str, optional
            The simulation function that was used to create the set of simulations.
        """

        if signals is None:
            return

        if isinstance(signals, Simulations):
            self.signals.append(signals)

        if isinstance(signals, list):

            if isinstance(signals[0], Simulations):
                self.signals.extend(signals)

            else:
                params = repeat(params) if not isinstance(params, list) else params
                sim_func = repeat(sim_func) if not isinstance(sim_func, list) else sim_func
                for csigs, cparams, cfunc in zip(signals, params, sim_func):
                    signals = Simulations(csigs, params=cparams, sim_func=cfunc)
                    self.signals.append(signals)
