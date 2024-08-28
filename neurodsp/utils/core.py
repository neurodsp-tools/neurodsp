"""Core / internal utility functions."""

from itertools import count
from collections.abc import Iterable

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


def listify(arg):
    """Check and embed an argument into a list, if is not already in a list.

    Parameters
    ----------
    arg : object
        Argument to check and embed in a list, if it is not already.

    Returns
    -------
    list
        Argument embedded in a list.
    """

    # Embed all non-iterable parameters into a list
    #   Note: deal with str as a special case of iterable that we want to embed
    if not isinstance(arg, Iterable) or isinstance(arg, str):
        out = [arg]
    # Deal with special case of multi dimensional numpy arrays - want to embed without flattening
    elif isinstance(arg, np.ndarray) and np.ndim(arg) > 1:
        out = [arg]
    # If is iterable (e.g. tuple or numpy array), typecast to list
    else:
        out = list(arg)

    return out
