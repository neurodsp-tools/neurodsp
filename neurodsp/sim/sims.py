"""Objects for managing groups of simulated signals."""

import numpy as np

from neurodsp.sim.utils import drop_base_params

###################################################################################################
###################################################################################################

class Simulations():
    """Data object for a set of simulated signals.

    Parameters
    ----------
    signals : 1d or 2nd array
        The simulated signals, organized as [n_sims, sig_length].
    func : str
        The simulation function that was used to create the simulations.
    params : dict
        The simulation parameters that was used to create the simulations.

    Notes
    -----
    This object stores a set of simulations generated from a shared parameter definition.
    """

    def __init__(self, signals=None, func=None, parameters=None):
        """Initialize Simulations object."""

        self.signals = np.atleast_2d(signals) if signals is not None else np.array([])
        self.func = func

        self.n_seconds = None
        self.fs = None
        self._params = {}
        if parameters is not None:
            self.add_params(parameters)

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
    def params(self):
        """Define the full set of simulation parameters (base + additional parameters)."""

        return {**self._base_params, **self._params}

    @property
    def _base_params(self):
        """Define the base parameters."""

        return {'n_seconds' : self.n_seconds, 'fs' : self.fs}

    def add_params(self, parameters):
        """Add parameter definition to object."""

        self.n_seconds = parameters['n_seconds']
        self.fs = parameters['fs']
        self._params = drop_base_params(parameters)


class SampledSimulations(Simulations):
    """Data object for a set of simulated signals with sampled (variable) parameter definitions.

    Parameters
    ----------
    signals : 2nd array
        The simulated signals, organized as [n_sims, sig_length].
    func : str
        The simulation function that was used to create the simulations.
    params : list dict
        The simulation parameters for each of the simulations.

    Notes
    -----
    This object stores a set of simulations with different parameter definitions per signal.
    """

    def __init__(self, signals=None, func=None, parameters=None):
        """Initialize VSims object."""

        self._params = []
        Simulations.__init__(self, signals, func, parameters)

    def add_params(self, parameters):
        """Add parameter definitions to object."""

        self.n_seconds = parameters[0]['n_seconds']
        self.fs = parameters[0]['fs']

        self._params = []
        for cparams in parameters:
            self._params.append(drop_base_params(cparams))

    @property
    def params(self):
        """Define simulation parameters (base + additional parameters) for each simulation."""

        return [{**self._base_params, **self._params[ind]} for ind in range(len(self))]

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
        if parameters:
            self._params.append(drop_base(parameters))
        else:
            assert len(self._params) == 0, 'Must add parameters if object already has them.'
