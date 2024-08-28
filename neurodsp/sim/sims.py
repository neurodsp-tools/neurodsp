"""Objects for managing groups of simulated signals."""

import numpy as np

from neurodsp.utils.core import listify
from neurodsp.sim.utils import get_base_params, drop_base_params

###################################################################################################
###################################################################################################

class Simulations():
    """Data object for a set of simulated signals.

    Parameters
    ----------
    signals : 1d or 2nd array, optional
        The simulated signals, organized as [n_sims, sig_length].
    params : dict, optional
        The simulation parameters that were used to create the simulations.
    sim_func : str, optional
        The simulation function that was used to create the simulations.

    Notes
    -----
    This object stores a set of simulations generated from a shared parameter definition.
    """

    def __init__(self, signals=None, params=None, sim_func=None):
        """Initialize Simulations object."""

        self.signals = np.atleast_2d(signals) if signals is not None else np.array([])
        self._base_params = None
        self._params = None
        self.add_params(params)
        self.sim_func = sim_func

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

        return {**self._base_params, **self._params}

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
        params : dict, optional
            The simulation parameter definition(s).
        """

        if params:
            self._base_params = get_base_params(params)
            self._params = drop_base_params(params)


class SampledSimulations(Simulations):
    """Data object for a set of simulated signals with sampled (variable) parameter definitions.

    Parameters
    ----------
    signals : 2nd array, optional
        The simulated signals, organized as [n_sims, sig_length].
    params : list of dict, optional
        The simulation parameters for each of the simulations.
    sim_func : str, optional
        The simulation function that was used to create the simulations.

    Notes
    -----
    This object stores a set of simulations with different parameter definitions per signal.
    """

    def __init__(self, signals=None, params=None, sim_func=None):
        """Initialize SampledSimulations object."""

        Simulations.__init__(self, signals, params, sim_func)

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

    def add_params(self, params):
        """Add parameter definition(s) to object.

        Parameters
        ----------
        params : dict or list of dict, optional
            The simulation parameter definition(s).
        """

        if params:

            params = listify(params)
            base_params = get_base_params(params[0])
            cparams = [drop_base_params(el) for el in params]

            if not self.has_params:
                if len(self) > len(cparams):
                   msg = 'Cannot add parameters to object without existing parameter values.'
                   raise ValueError(msg)
                self._base_params = base_params
                self._params = cparams

            else:
                self._params.extend(cparams)

        else:
            if self.has_params:
                raise ValueError('Must add parameters if object already has them.')

    def add_signal(self, signal, params=None):
        """Add a signal to the current object.

        Parameters
        ----------
        signal : 1d array
            A simulated signal to add to the object.
        params : dict, optional
            Parameter definition for the added signal.
            If current object does not include parameters, should be empty.
            If current object does include parameters, this input is required.
        """

        try:
            self.signals = np.vstack([self.signals, signal])
        except ValueError:
            raise ValueError('Size of the added signal is not consistent with existing signals.')
        self.add_params(params)
