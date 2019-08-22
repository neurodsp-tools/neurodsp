"""Core / internal utility functions."""

from itertools import repeat

import numpy as np

###################################################################################################
###################################################################################################

def get_avg_func(avg_type):
    """Select a function to use for averaging.

    Parameters
    ----------
    avg_type : {'mean', 'median'}
        The type of averaging function to use.

    Returns
    -------
    avg_func : callable
        Requested averaging function.
    """

    if avg_type == 'mean':
        avg_func = np.mean
    elif avg_type == 'median':
        avg_func = np.median
    else:
        raise ValueError('Averaging method not understood.')

    return avg_func


def check_n_cycles(n_cycles, len_cycles=None):
    """Check an input as a number of cycles definition, and make it iterable.

    Parameters
    ----------
    n_cycles : float or list
        Definiton of number of cycles.
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
