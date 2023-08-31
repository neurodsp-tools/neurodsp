"""Plotting functions for plots with combined panels."""

import matplotlib.pyplot as plt

from neurodsp.spectral import compute_spectrum
from neurodsp.spectral.utils import trim_spectrum
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series
from neurodsp.plts.utils import savefig

from neurodsp.utils.checks import check_dict_key

###################################################################################################
###################################################################################################

@savefig
def plot_timeseries_and_spectrum(times, sig, fs, f_range=None, spectrum_kwargs=None,
                                 ts_kwargs=None, psd_kwargs=None, **plt_kwargs):
    """Plot a timeseries together with it's associated power spectrum.

    Parameters
    ----------
    times : 1d array, or None
        Time definition(s) for the time series to be plotted.
        If None, time series will be plotted in terms of samples instead of time.
    sigs : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    f_range : list of [float, float], optional
        The frequency range to restrict the power spectrum to.
    **spectrum_kwargs : dict, optional
        Keyword arguments for computing the power spectrum.
        See `compute_spectrum` for details.
    **ts_kwargs : dict, optional
        Keyword arguments for customizing the time series plot.
    **psd_kwargs : dict, optional
        Keyword arguments for customizing the power spectrum plot.
    **plt_kwargs
        Keyword arguments for customizing the plots.
        These arguments are passed to both plot axes.
    """

    # Allow for defining color as 'color' (since one line per plot), rather than 'colors'
    if 'color' in plt_kwargs:
        plt_kwargs['colors'] = plt_kwargs.pop('color')

    # Default to drawing both plots in same color (otherwise ts is black, psd is blue)
    if 'colors' not in plt_kwargs:
        psd_kwargs = {} if psd_kwargs is None else psd_kwargs
        if 'colors' not in psd_kwargs:
            psd_kwargs['colors'] = 'black'

    fig = plt.figure(figsize=plt_kwargs.pop('figsize', None))
    ax1 = fig.add_axes([0.0, 0.6, 1.3, 0.5])
    ax2 = fig.add_axes([1.5, 0.6, 0.6, 0.5])

    plot_time_series(times, sig, ax=ax1, **plt_kwargs,
                     **ts_kwargs if ts_kwargs else {})

    freqs, psd = compute_spectrum(sig, fs, **spectrum_kwargs if spectrum_kwargs else {})
    if f_range:
        freqs, psd = trim_spectrum(freqs, psd, f_range)
    plot_power_spectra(freqs, psd, ax=ax2, **plt_kwargs,
                       **psd_kwargs if psd_kwargs else {})
