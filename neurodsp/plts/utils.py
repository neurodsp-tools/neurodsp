"""Utility functions for plots."""

from functools import wraps
from os.path import join as pjoin

import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

def check_ax(ax, figsize=None):
    """Check whether a figure axes object is defined, define if not.

    Parameters
    ----------
    ax : matplotlib.Axes or None
        Axes object to check if is defined.

    Returns
    -------
    ax : matplotlib.Axes
        Figure axes object to use.
    """

    if not ax:
        _, ax = plt.subplots(figsize=figsize)

    return ax


def savefig(func):
    """Decorator function to save out figures."""

    @wraps(func)
    def decorated(*args, **kwargs):

        save_fig = kwargs.pop('save_fig', False)
        file_name = kwargs.pop('file_name', None)
        file_path = kwargs.pop('file_path', None)

        func(*args, **kwargs)

        if save_fig:
            full_path = pjoin(file_path, file_name) if file_path else file_name
            plt.savefig(full_path)

    return decorated
