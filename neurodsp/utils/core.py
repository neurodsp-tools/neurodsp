"""Core / internal utility functions."""

from itertools import count

import numpy as np

from neurodsp.utils.checks import check_param_options

###################################################################################################
###################################################################################################

def get_avg_func(avg_type):
    """Select a function to use for averaging.

    Parameters
    ----------
    avg_type : {'mean', 'median', 'sum'}
        The type of averaging function to use.

    Returns
    -------
    func : callable
        Requested function.
    """

    check_param_options(avg_type, 'avg_type', ['mean', 'median', 'sum'])

    if avg_type == 'mean':
        func = np.mean
    elif avg_type == 'median':
        func = np.median
    elif avg_type == 'sum':
        func = np.sum

    return func


def counter(value):
    """Counter that supports both finite and infinite ranges.

    Parameters
    ----------
    value : int or None
        Upper bound for the counter (if finite) or None (if infinite).

    Returns
    -------
    counter : range or count
        Counter object for finite (range) or infinite (count) iteration.
    """

    return range(value) if value else count()
