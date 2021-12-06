"""Core / internal utility functions."""

import numpy as np

from neurodsp.utils.checks import check_param_options

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

    check_param_options(avg_type, 'avg_type', ['mean', 'median'])

    if avg_type == 'mean':
        avg_func = np.mean
    elif avg_type == 'median':
        avg_func = np.median

    return avg_func
