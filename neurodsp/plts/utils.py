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

        # Grab file name and path arguments, if they are in kwargs
        file_name = kwargs.pop('file_name', None)
        file_path = kwargs.pop('file_path', None)

        # Check for an explicit argument for whether to save figure or not
        #   Defaults to saving when file name given (since bool(str)->True; bool(None)->False)
        save_fig = kwargs.pop('save_fig', bool(file_name))

        # Check any collect any other plot keywords
        save_kwargs = kwargs.pop('save_kwargs', {})
        save_kwargs.setdefault('bbox_inches', 'tight')

        # Check and collect whether to close the plot
        close = kwargs.pop('close', False)

        func(*args, **kwargs)

        if save_fig:
            save_figure(file_name, file_path, close, **save_kwargs)

    return decorated


def save_figure(file_name, file_path=None, close=False, **save_kwargs):
    """Save out a figure.

    Parameters
    ----------
    file_name : str
        File name for the figure file to save out.
    file_path : str or Path
        Path for where to save out the figure to.
    close : bool, optional, default: False
        Whether to close the plot after saving.
    save_kwargs
        Additional arguments to pass into the save function.
    """

    full_path = pjoin(file_path, file_name) if file_path else file_name
    plt.savefig(full_path, **save_kwargs)

    if close:
        plt.close()
