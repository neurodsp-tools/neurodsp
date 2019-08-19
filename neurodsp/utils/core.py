"""Core / internal utility functions."""

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
