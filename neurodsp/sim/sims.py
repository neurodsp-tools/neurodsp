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
    signals : 1d or 2nd array
        The simulated signals, organized as [n_sims, sig_length].
    sim_func : str
        The simulation function that was used to create the simulations.
    params : dict
        The simulation parameters that were used to create the simulations.

    Notes
    -----
    This object stores a set of simulations generated from a shared parameter definition.
    """

    def __init__(self, signals=None, sim_func=None, params=None):
        """Initialize Simulations object."""

        self.signals = np.atleast_2d(signals) if signals is not None else np.array([])
        self.sim_func = sim_func

        self._base_params = None
        self._params = None
        self.add_params(params)

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
        return self._base_params['n_seconds'] if self.has_params else None

    @property
    def fs(self):
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
    signals : 2nd array
        The simulated signals, organized as [n_sims, sig_length].
    sim_func : str
        The simulation function that was used to create the simulations.
    params : list of dict
        The simulation parameters for each of the simulations.

    Notes
    -----
    This object stores a set of simulations with different parameter definitions per signal.
    """

    def __init__(self, signals=None, sim_func=None, params=None):
        """Initialize SampledSimulations object."""

        Simulations.__init__(self, signals, sim_func, params)

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
                self._base_params = base_params
                self._params = cparams

            else:
                assert get_base_params(params[0]) == self._base_params, \
                    'Base parameters must match the existing simulations in the object.'
                self._params.extend(cparams)

        else:
            assert not self._params, 'Must add parameters if object already has them.'

    def add_signal(self, sig, params=None):
        """Add a signal to the current object.

        Parameters
        ----------
        sig : 1d array
            A simulated signal to add to the object.
        params : dict, optional
            Parameter definition for the added signal.
            If current object does not include parameters, should be empty.
            If current object does include parameters, this input is required.
        """

        self.signals = np.vstack([self.signals, sig])
        self.add_params(params)
