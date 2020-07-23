"""Checker functions."""

from itertools import repeat

import numpy as np

###################################################################################################
###################################################################################################

def check_param(param, label, bounds):
    """Check a parameter value.

    Parameters
    ----------
    param : float
        Parameter value to check.
    label : str
        Label of the parameter being checked.
    bounds : list of [float, float]
       Bounding range of valid values for the given parameter.

    Raises
    ------
    ValueError
        If a parameter that is being checked is out of range.
    """

    if (param < bounds[0]) or (param > bounds[1]):
        msg = "The provided value for the {} parameter is out of bounds. ".format(label) + \
        "It should be between {:1.1f} and {:1.1f}.".format(*bounds)
        raise ValueError(msg)


def check_n_cycles(n_cycles, len_cycles=None):
    """Check an input as a number of cycles definition, and make it iterable.

    Parameters
    ----------
    n_cycles : float or list
        Definition of number of cycles.
        If a single value, the same number of cycles is used for each frequency value.
        If a list or list_like, then should be a n_cycles corresponding to each frequency.
    len_cycles : int, optional
        What the length of `n_cycles` should, if it's a list.

    Returns
    -------
    n_cycles : iterable
        An iterable version of the number of cycles.
    """

    if isinstance(n_cycles, (int, float)):

        if n_cycles <= 0:
            raise ValueError('Number of cycles must be a positive number.')

        n_cycles = repeat(n_cycles)

    elif isinstance(n_cycles, (list, np.ndarray)):

        for cycle in n_cycles:
            if cycle <= 0:
                raise ValueError('Each number of cycles must be a positive number.')

        if len_cycles and len(n_cycles) != len_cycles:
            raise ValueError('The length of number of cycles does not match other inputs.')

        n_cycles = iter(n_cycles)

    return n_cycles
