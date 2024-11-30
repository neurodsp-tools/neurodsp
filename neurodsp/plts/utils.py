"""Utility functions for plots."""

from copy import deepcopy
from functools import wraps
from os.path import join as pjoin
from itertools import repeat, cycle

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.plts.settings import SUPTITLE_FONTSIZE

###################################################################################################
###################################################################################################

def subset_kwargs(kwargs, label):
    """Subset a set of kwargs from a dictionary.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments.
    label : str
        Label to use to subset.
        Any entries with label in the key will be subset from the kwargs dict.

    Returns
    -------
    kwargs : dict
        The kwargs dictionary, with subset items removed.
    subset : dict
        The collection of subset kwargs.
    """

    kwargs = deepcopy(kwargs)

    subset = {}
    for key in list(kwargs.keys()):
        if label in key:
            subset[key] = kwargs.pop(key)

    return kwargs, subset


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


def make_axes(n_rows, n_cols, figsize=None, row_size=4, col_size=3.6,
              wspace=None, hspace=None, title=None, **plt_kwargs):
    """Make a subplot with multiple axes.

    Parameters
    ----------
    n_rows, n_cols : int
        The number of rows and columns axes to create in the figure.
    figsize : tuple of float, optional
        Size to make the overall figure.
        If not given, is estimated from the number of axes.
    row_size, col_size : float, optional
        The size to use per row / column.
        Only used if `figsize` is None.
    wspace, hspace : float, optional
        Parameters for spacing between subplots.
        These get passed into `plt.subplots_adjust`.
    title : str, optional
        A super title to add to the figure.
    **plt_kwargs
        Extra arguments to pass to `plt.subplots`.

    Returns
    -------
    axes : 1d array of AxesSubplot
        Collection of axes objects.
    """

    if not figsize:
        figsize = (n_cols * col_size, n_rows * row_size)

    plt_kwargs, title_kwargs = subset_kwargs(plt_kwargs, 'title')

    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize, **plt_kwargs)

    if wspace or hspace:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)

    if title:
        plt.suptitle(title,
                     fontsize=title_kwargs.pop('title_fontsize', SUPTITLE_FONTSIZE),
                     **title_kwargs)

    return axes


def prepare_multi_plot(xs, ys, labels=None, colors=None):
    """Prepare inputs for plotting one or more elements in a loop.

    Parameters
    ----------
    xs, ys : 1d or 2d array
        Plot data.
    labels : str or list
        Label(s) for the plot input(s).
    colors : str or iterable
        Color(s) to plot input(s).

    Returns
    -------
    xs, ys : iterable
        Plot data.
    labels : iterable
        Label(s) for the plot input(s).
    colors : iterable
        Color(s) to plot input(s).

    Notes
    -----
    This function takes inputs that can reflect one or more plot elements, and
    prepares the inputs to be iterable for plotting in a loop.
    """

    xs = repeat(xs) if isinstance(xs, np.ndarray) and xs.ndim == 1 else xs
    ys = [ys] if isinstance(ys, np.ndarray) and ys.ndim == 1 else ys

    if labels is not None:
        labels = [labels] if not isinstance(labels, list) else labels
    else:
        labels = repeat(labels)

    colors = repeat(colors) if not isinstance(colors, list) else cycle(colors)

    return xs, ys, labels, colors
