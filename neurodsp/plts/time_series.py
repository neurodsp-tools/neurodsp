"""Plots for time series."""

from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.plts.style import style_plot
from neurodsp.plts.utils import check_ax, savefig, prepare_multi_plot
from neurodsp.utils.data import create_samples
from neurodsp.utils.checks import check_param_options

###################################################################################################
###################################################################################################

@savefig
@style_plot
def plot_time_series(times, sigs, labels=None, colors=None, ax=None, **kwargs):
    """Plot a time series.

    Parameters
    ----------
    times : 1d or 2d array, or list of 1d array, or None
        Time definition(s) for the time series to be plotted.
        If None, time series will be plotted in terms of samples instead of time.
    sigs : 1d or 2d array, or list of 1d array
        Time series to plot.
    labels : list of str, optional
        Labels for each time series.
    colors : str or list of str
        Color(s) to use to plot lines.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments for customizing the plot.

    Examples
    --------
    Create a time series plot:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.utils import create_times
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {'exponent': -1.5, 'f_range': (2, None)},
    ...                                'sim_oscillation' : {'freq': 10}})
    >>> times = create_times(n_seconds=10, fs=500)
    >>> plot_time_series(times, sig)
    """

    ax = check_ax(ax, kwargs.pop('figsize', (15, 3)))

    times, xlabel = _check_times(times, sigs)
    times, sigs, colors, labels = prepare_multi_plot(times, sigs, colors, labels)

    # If not provided, default colors for up to two signals to be black & red
    if isinstance(colors, repeat) and next(colors) is None and len(sigs) <= 2:
        colors = ['k', 'r']

    for time, sig, color, label in zip(times, sigs, colors, labels):
        ax.plot(time, sig, color=color, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Voltage (uV)')


@savefig
@style_plot
def plot_instantaneous_measure(times, sigs, measure='phase', ax=None, **kwargs):
    """Plot an instantaneous measure, of phase, amplitude or frequency.

    Parameters
    ----------
    times : 1d or 2d array, or list of 1d array, or None
        Time definition(s) for the time series to be plotted.
        If None, time series will be plotted in terms of samples instead of time.
    sigs : 1d or 2d array, or list of 1d array
        Time series to plot.
    measure : {'phase', 'amplitude', 'frequency'}
        Which kind of measure is being plotted.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments to pass into `plot_time_series`, and/or for customizing the plot.

    Examples
    --------
    Create an instantaneous phase plot:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.utils import create_times
    >>> from neurodsp.timefrequency import phase_by_time
    >>> sig = sim_combined(n_seconds=2, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> pha = phase_by_time(sig, fs=500, f_range=(8, 12))
    >>> times = create_times(n_seconds=2, fs=500)
    >>> plot_instantaneous_measure(times, pha, measure='phase')
    """

    check_param_options(measure, 'measure', ['phase', 'amplitude', 'frequency'])

    if measure == 'phase':
        plot_time_series(times, sigs, ax=ax, ylabel='Phase (rad)', **kwargs)
        ax = ax if ax else plt.gca()
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels([r'-$\pi$', 0, r'$\pi$'])
    elif measure == 'amplitude':
        plot_time_series(times, sigs, ax=ax, ylabel='Amplitude', **kwargs)
    elif measure == 'frequency':
        plot_time_series(times, sigs, ax=ax, ylabel='Instantaneous\nFrequency (Hz)', **kwargs)


@savefig
@style_plot
def plot_bursts(times, sig, bursting, ax=None, **kwargs):
    """Plot a time series, with labeled bursts.

    Parameters
    ----------
    times : 1d array or None
        Time definition for the time series to be plotted.
        If None, time series will be plotted in terms of samples instead of time.
    sig : 1d array
        Time series to plot.
    bursting : 1d array
        A boolean array which indicates identified bursts.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments to pass into `plot_time_series`, and/or for customizing the plot.

    Examples
    --------
    Create a plot of burst activity:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.utils import create_times
    >>> from neurodsp.burst import detect_bursts_dual_threshold
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_synaptic_current': {},
    ...                                'sim_bursty_oscillation' : {'freq': 10}},
    ...                    component_variances=(0.1, 0.9))
    >>> is_burst = detect_bursts_dual_threshold(sig, fs=500, dual_thresh=(1, 2), f_range=(8, 12))
    >>> times = create_times(n_seconds=10, fs=500)
    >>> plot_bursts(times, sig, is_burst, labels=['Raw Data', 'Detected Bursts'])
    """

    bursts = np.ma.array(sig, mask=np.invert(bursting))
    plot_time_series(times, [sig, bursts], ax=ax, **kwargs)


@savefig
@style_plot
def plot_multi_time_series(times, sigs, colors=None, ax=None, **plt_kwargs):
    """Plot multiple time series, with a vertical offset.

    Parameters
    ----------
    times : 1d or 2d array, or list of 1d array, or None
        Time definition(s) for the time series to be plotted.
        If None, time series will be plotted in terms of samples instead of time.
    sigs : 2d array or list of 1d array
        Time series to plot, each list or row of the array representing a different channel.
    colors : str or list of str
        Color(s) to use to plot lines.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments for customizing the plot.
    """

    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize',  (15, 5)))

    colors = 'black' if not colors else colors

    times, xlabel = _check_times(times, sigs)
    times, sigs, _, colors = prepare_multi_plot(times, sigs, None, colors)

    step = 0.8 * np.ptp(sigs[0])

    for ind, (time, sig) in enumerate(zip(times, sigs)):
        ax.plot(time, sig+step*ind, color=next(colors), **plt_kwargs)

    ax.set(xlabel=xlabel, ylabel='Channels', yticks=[])


def _check_times(times, sigs):
    """Helper function to check a times definition passed into a time series plot function."""

    xlabel = 'Time (s)'
    if times is None:
        if isinstance(sigs, list) or (isinstance(sigs, np.ndarray) and sigs.ndim == 2):
            n_samples = len(sigs[0])
        else:
            n_samples = len(sigs)
        times = create_samples(n_samples)
        xlabel = 'Samples'

    return times, xlabel
