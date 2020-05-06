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

    Examples
    --------
    Plot the average pattern from a sliding window matching analysis:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.rhythm import sliding_window_matching
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_synaptic_current': {},
    ...                                'sim_bursty_oscillation' : {'freq': 20,
    ...                                                            'enter_burst': .10,
    ...                                                            'leave_burst': .15}},
    ...                    component_variances=(0.001, 0.90))
    >>> avg_window, _, _ = sliding_window_matching(sig, fs=500, win_len=0.05, win_spacing=0.5)
    >>> plot_swm_pattern(avg_window)

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

    Examples
    --------
    Plot lagged coherence:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.rhythm import compute_lagged_coherence
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_synaptic_current': {},
    ...                                'sim_bursty_oscillation' : {'freq': 20,
    ...                                                            'enter_burst': .10,
    ...                                                            'leave_burst': .15}},
    ...                    component_variances=(0.001, 0.90))
    >>> lag_coh_beta, freqs = compute_lagged_coherence(sig, fs=500, freqs=(15, 30),
    ...                                                return_spectrum=True)
    >>> plot_lagged_coherence(freqs, lag_coh_beta)

    """

    ax = check_ax(ax, (6, 3))

    plt.plot(freqs, lcs, 'k.-')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Lagged Coherence')
