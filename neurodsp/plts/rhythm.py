"""Plotting functions for neurodsp.rhythm."""

import matplotlib.pyplot as plt

from neurodsp.plts.style import style_plot
from neurodsp.plts.utils import check_ax, savefig

###################################################################################################
###################################################################################################

@savefig
@style_plot
def plot_swm_pattern(pattern, ax=None):
    """Plot the resulting pattern from a sliding window matching analysis.

    Parameters
    ----------
    pattern : 1d array
        The resulting average pattern from applying sliding window matching.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    """

    ax = check_ax(ax, (4, 4))

    plt.plot(pattern, 'k')

    plt.title('Average Pattern')
    plt.xlabel('Time (samples)')
    plt.ylabel('Voltage (a.u.)')


@savefig
@style_plot
def plot_lagged_coherence(freqs, lcs, ax=None):
    """Plot lagged coherence values across frequencies.

    Parameters
    ----------
    freqs : 1d array
        Vector of frequencies at which lagged coherence was computed.
    lcs : 1d array
        Lagged coherence values across the computed frequencies.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    """

    ax = check_ax(ax, (6, 3))

    plt.plot(freqs, lcs, 'k.-')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Lagged Coherence')
