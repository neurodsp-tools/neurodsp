"""Plotting functions for neurodsp.timefrequency."""

import numpy as np

from neurodsp.plts.style import style_plot
from neurodsp.plts.utils import check_ax, savefig

###################################################################################################
###################################################################################################

@savefig
@style_plot
def plot_timefrequency(times, freqs, powers, x_ticks=5, y_ticks=5, ax=None, **kwargs):
    """Plot a time-frequency representation of data.

    Parameters
    ----------
    times : 1d array
        The time dimension for the time-frequency representation.
    freqs : 1d array
        The frequency dimension for the time-frequency representation.
    powers : 2d array
        Power values to plot.
        If array is complex, the real component is taken for plotting.
    x_ticks, y_ticks : int or array_like
        Defines the tick labels to add to the plot.
        If int, is the number of evenly sampled labels to add to the plot.
        If array_like, is a set of labels to add to the plot.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments for customizing the plot.

    Examples
    --------
    Plot a Morlet transformation:

    >>> import numpy as np
    >>> from neurodsp.sim import sim_bursty_oscillation
    >>> from neurodsp.timefrequency.wavelets import compute_wavelet_transform
    >>> fs=1000
    >>> sig = sim_bursty_oscillation(n_seconds=10, fs=fs, freq=10)
    >>> times = np.arange(0, len(sig)/fs, 1/fs)
    >>> freqs = np.arange(1, 50, 1)
    >>> mwt = compute_wavelet_transform(sig, fs, freqs)
    >>> plot_timefrequency(times, freqs, mwt)
    """

    ax = check_ax(ax, figsize=kwargs.pop('figsize', None))

    if np.iscomplexobj(powers):
        powers = abs(powers)

    ax.imshow(powers, aspect='auto', **kwargs)
    ax.invert_yaxis()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    if isinstance(x_ticks, int):
        x_tick_pos = np.linspace(0, times.size, x_ticks)
        x_ticks = np.round(np.linspace(times[0], times[-1], x_ticks), 2)
    else:
        x_tick_pos = [np.argmin(np.abs(times - val)) for val in x_ticks]
    ax.set(xticks=x_tick_pos, xticklabels=x_ticks)

    if isinstance(y_ticks, int):
        y_ticks_pos = np.linspace(0, freqs.size, y_ticks)
        y_ticks = np.round(np.linspace(freqs[0], freqs[-1], y_ticks), 2)
    else:
        y_ticks_pos = [np.argmin(np.abs(freqs - val)) for val in y_ticks]
    ax.set(yticks=y_ticks_pos, yticklabels=y_ticks)
