"""Plotting functions for neurodsp.rhythm."""

from neurodsp.plts.style import style_plot
from neurodsp.plts.utils import check_ax, savefig

###################################################################################################
###################################################################################################

@savefig
@style_plot
def plot_swm_pattern(pattern, ax=None, **kwargs):
    """Plot the resulting pattern from a sliding window matching analysis.

    Parameters
    ----------
    pattern : 1d array
        The resulting average pattern from applying sliding window matching.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments for customizing the plot.

    Examples
    --------
    Plot the average pattern from a sliding window matching analysis:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.rhythm import sliding_window_matching
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {'f_range': (2, None)},
    ...                                'sim_bursty_oscillation': {'freq': 20,
    ...                                                           'enter_burst': .25,
    ...                                                           'leave_burst': .25}})
    >>> avg_window, _, _ = sliding_window_matching(sig, fs=500, win_len=0.05, win_spacing=0.5)
    >>> plot_swm_pattern(avg_window)
    """

    ax = check_ax(ax, (4, 4))

    ax.plot(pattern, 'k')

    ax.set_title('Average Pattern')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Voltage (a.u.)')


@savefig
@style_plot
def plot_lagged_coherence(freqs, lcs, ax=None, **kwargs):
    """Plot lagged coherence values across frequencies.

    Parameters
    ----------
    freqs : 1d array
        Vector of frequencies at which lagged coherence was computed.
    lcs : 1d array
        Lagged coherence values across the computed frequencies.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments for customizing the plot.

    Examples
    --------
    Plot lagged coherence:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.rhythm import compute_lagged_coherence
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_synaptic_current': {},
    ...                                'sim_bursty_oscillation': {'freq': 20,
    ...                                                           'enter_burst': .50,
    ...                                                           'leave_burst': .25}})
    >>> lag_cohs, freqs = compute_lagged_coherence(sig, fs=500, freqs=(5, 35),
    ...                                            return_spectrum=True)
    >>> plot_lagged_coherence(freqs, lag_cohs)
    """

    ax = check_ax(ax, (6, 3))

    ax.plot(freqs, lcs, 'k.-')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Lagged Coherence')
