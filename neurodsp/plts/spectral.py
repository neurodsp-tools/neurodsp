"""Plotting functions for neurodsp.spectral."""

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.plts.style import style_plot
from neurodsp.plts.utils import check_ax, check_ax_3d, savefig, prepare_multi_plot

###################################################################################################
###################################################################################################

@savefig
@style_plot
def plot_power_spectra(freqs, powers, labels=None, colors=None, ax=None, **kwargs):
    """Plot power spectra.

    Parameters
    ----------
    freqs : 1d or 2d array or list of 1d array
        Frequency vector.
    powers : 1d or 2d array or list of 1d array
        Power values.
    labels : str or list of str, optional
        Labels for each time series.
    colors : str or list of str
        Colors to use to plot lines.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments for customizing the plot.

    Examples
    --------
    Plot a power spectrum:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.spectral import compute_spectrum
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_synaptic_current': {},
    ...                                'sim_bursty_oscillation' : {'freq': 10}},
    ...                    component_variances=(0.5, 1.))
    >>> freqs, powers = compute_spectrum(sig, fs=500)
    >>> plot_power_spectra(freqs, powers)
    """

    ax = check_ax(ax, figsize=kwargs.pop('figsize', (6, 6)))

    for freq, power, label, color in zip(*prepare_multi_plot(freqs, powers, labels, colors)):
        ax.loglog(freq, power, label=label, color=color)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power ($V^2/Hz$)')


@savefig
@style_plot
def plot_scv(freqs, scv, ax=None, **kwargs):
    """Plot spectral coefficient of variation.

    Parameters
    ----------
    freqs : 1d array
        Frequency vector.
    scv : 1d array
        Spectral coefficient of variation.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments for customizing the plot.

    Examples
    --------
    Plot the spectral coefficient of variation:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.spectral import compute_scv
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs, scv = compute_scv(sig, fs=500)
    >>> plot_scv(freqs, scv)
    """

    ax = check_ax(ax, figsize=kwargs.pop('figsize', (5, 5)))

    ax.loglog(freqs, scv)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('SCV')


@savefig
@style_plot
def plot_scv_rs_lines(freqs, scv_rs, ax=None, **kwargs):
    """Plot spectral coefficient of variation, from the resampling method, as lines.

    Parameters
    ----------
    freqs : 1d array
        Frequency vector.
    scv_rs : 2d array
        Spectral coefficient of variation, from resampling procedure.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments for customizing the plot.

    Examples
    --------
    Plot the spectral coefficient of variation using a resampling method:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.spectral import compute_scv_rs
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs, t_inds, scv_rs = compute_scv_rs(sig, fs=500, nperseg=500, method='bootstrap',
    ...                                        rs_params=(5, 200))
    >>> plot_scv_rs_lines(freqs, scv_rs)
    """

    ax = check_ax(ax, figsize=kwargs.pop('figsize', (8, 8)))

    ax.loglog(freqs, scv_rs, 'k', alpha=0.1)
    ax.loglog(freqs, np.mean(scv_rs, axis=1), lw=2)
    ax.loglog(freqs, len(freqs)*[1.])

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('SCV')


@savefig
@style_plot
def plot_scv_rs_matrix(freqs, t_inds, scv_rs, ax=None, **kwargs):
    """Plot spectral coefficient of variation, from the resampling method, as a matrix.

    Parameters
    ----------
    freqs : 1d array
        Frequency vector.
    t_inds : 1d array
        Time indices.
    scv_rs : 1d array
        Spectral coefficient of variation, from resampling procedure.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments for customizing the plot.

    Examples
    --------
    Plot a SCV matrix from a simulated signal with a high probability of bursting at 10Hz:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.spectral import compute_scv_rs
    >>> sig = sim_combined(n_seconds=100, fs=500,
    ...                    components={'sim_synaptic_current': {},
    ...                                'sim_bursty_oscillation': {'freq': 10, 'enter_burst':0.75}})
    >>> freqs, t_inds, scv_rs = compute_scv_rs(sig, fs=500, method='rolling', rs_params=(10, 2))
    >>> # Plot the computed scv, plotting frequencies up to 20 Hz (index of 21)
    >>> plot_scv_rs_matrix(freqs[:21], t_inds, scv_rs[:21])
    """

    ax = check_ax(ax, figsize=kwargs.pop('figsize', (10, 5)))

    im = ax.imshow(np.log10(scv_rs), aspect='auto',
                   extent=(t_inds[0], t_inds[-1], freqs[-1], freqs[0]))
    plt.colorbar(im, label='SCV')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')


@savefig
@style_plot
def plot_spectral_hist(freqs, power_bins, spectral_hist, spectrum_freqs=None,
                       spectrum=None, ax=None, **kwargs):
    """Plot spectral histogram.

    Parameters
    ----------
    freqs : 1d array
        Frequencies over which the histogram is calculated.
    power_bins : 1d array
        Power bins within which histogram is aggregated.
    spectral_hist : 2d array
        Spectral histogram to be plotted.
    spectrum_freqs : 1d array, optional
        Frequency axis of the power spectrum to be plotted.
    spectrum : 1d array, optional
        Spectrum to be plotted over the histograms.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **kwargs
        Keyword arguments for customizing the plot.

    Examples
    --------
    Plot a spectral histogram:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.spectral import compute_spectral_hist
    >>> sig = sim_combined(n_seconds=100, fs=500,
    ...                    components={'sim_synaptic_current': {},
    ...                                'sim_bursty_oscillation' : {'freq': 10}},
    ...                    component_variances=(0.5, 1))
    >>> freqs, bins, spect_hist = compute_spectral_hist(sig, fs=500, nbins=40, f_range=(1, 75),
    ...                                                 cut_pct=(0.1, 99.9))
    >>> plot_spectral_hist(freqs, bins, spect_hist)
    """

    # Get axis, with default of scaling figure height based on number of bins
    figsize = (8, 12 * len(power_bins) / len(freqs))
    ax = check_ax(ax, figsize=kwargs.pop('figsize', figsize))

    # Plot histogram intensity as image and automatically adjust aspect ratio
    im = ax.imshow(spectral_hist, extent=[freqs[0], freqs[-1], power_bins[0], power_bins[-1]],
                   aspect='auto')
    plt.colorbar(im, label='Probability')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Log10 Power')

    # If a power spectrum is provided, plot over the histogram data
    if spectrum is not None:
        plt_inds = np.logical_and(spectrum_freqs >= freqs[0], spectrum_freqs <= freqs[-1])
        ax.plot(spectrum_freqs[plt_inds], np.log10(spectrum[plt_inds]), color='w', alpha=0.8)


@savefig
@style_plot
def plot_spectra_3d(freqs, powers, log_freqs=False, log_powers=True, colors=None,
                    orientation=(20, -50), zoom=1.0, ax=None, **kwargs):
    """Plot a series of power spectra in a 3D plot.

    Parameters
    ----------
    freqs : 1d or 2d array or list of 1d array
        Frequency vector.
    powers : 2d array or list of 1d array
        Power values.
    log_freqs : bool, optional, default: False
        Whether to plot the frequency values in log10 space.
    log_powers : bool, optional, default: True
        Whether to plot the power values in log10 space.
    colors : str or list of str, optional
        Colors to use to plot lines.
    orientation : tuple of int, optional, default: (20, -50)
        Orientation to set the 3D plot. See `Axes3D.view_init` for more information.
    zoom : float, optional, default: 1.0
        Zoom scaling for the figure axis. See `Axes3D.set_box_aspect` for more information.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot. Must be a 3D axis.
    **kwargs
        Keyword arguments for customizing the plot.

    Examples
    --------
    Plot power spectra in 3D:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.spectral import compute_spectrum
    >>> sig1 = sim_combined(n_seconds=10, fs=500,
    ...                     components={'sim_powerlaw': {'exponent' : -1},
    ...                                 'sim_bursty_oscillation' : {'freq': 10}})
    >>> sig2 = sim_combined(n_seconds=10, fs=500,
    ...                     components={'sim_powerlaw': {'exponent' : -1.5},
    ...                                 'sim_bursty_oscillation' : {'freq': 10}})
    >>> freqs1, powers1 = compute_spectrum(sig1, fs=500)
    >>> freqs2, powers2 = compute_spectrum(sig2, fs=500)
    >>> plot_spectra_3d([freqs1, freqs2], [powers1, powers2])
    """

    ax = check_ax_3d(ax)

    n_spectra = len(powers)

    for ind, (freq, power, _, color) in \
        enumerate(zip(*prepare_multi_plot(freqs, powers, None, colors))):
        ax.plot(xs=np.log10(freq) if log_freqs else freq,
                ys=[ind] * len(freq),
                zs=np.log10(power) if log_powers else power,
                color=color,
                **kwargs)

    ax.set(
        xlabel='Frequency (Hz)',
        ylabel='Channels',
        zlabel='Power',
        ylim=[0, n_spectra - 1],
    )

    yticks = list(range(n_spectra))
    ax.set_yticks(yticks, yticks)

    ax.view_init(*orientation)
    ax.set_box_aspect(None, zoom=zoom)
