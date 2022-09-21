"""Core / internal utility functions."""

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
