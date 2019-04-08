"""Plots for time series, for NeuroDSP."""

from itertools import repeat

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

def plot_time_series(times, sig, labels=None, ax=None):
    """Plot a neural time series.

    Parameters
    ----------
    times : 1d array or list of 1d array
        Time definition(s) for the time series to be plotted.
    sig : 1d array or list of 1d array
        Time series to plot.
    labels : list of str, optional
        Labels for each time series.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    """

    ax = check_ax(ax, (15, 3))

    times = repeat(times) if isinstance(times, np.ndarray) else times
    sigs = repeat(sig) if isinstance(sig, np.ndarray) else sig
    labels = repeat(labels) if not isinstance(labels, list) else labels
    cols = ['k', 'r', 'b']

    for time, sig, col, label in zip(times, sigs, cols, labels):
        plt.plot(time, sig, col, label=label)

    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (uV)')


def plot_bursts(times, sig, bursting, labels=None, ax=None):
    """Plot a neural time series, labelling detected bursts.

    Parameters
    ----------
    times : 1d array
        Time definition for the time series to be plotted.
    sig : 1d array
        Time series to plot.
    bursting : 1d array
        A boolean array which indicates identified bursts.
    labels : list of str, optional
        Labels for each time series.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    """

    ax = check_ax(ax, (15, 3))

    bursts = ma.array(sig, mask=np.invert(bursting))
    plot_time_series(times, [sig, bursts], labels=labels, ax=ax)
