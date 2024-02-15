"""Plotting functions for plots with combined panels."""

import matplotlib.pyplot as plt

from neurodsp.spectral import compute_spectrum
from neurodsp.utils.data import create_times
from neurodsp.spectral.utils import trim_spectrum
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series
from neurodsp.plts.utils import savefig

###################################################################################################
###################################################################################################

@savefig
def plot_timeseries_and_spectrum(sig, fs, ts_range=None, f_range=None, spectrum_kwargs=None,
                                 start_val=0., ts_kwargs=None, psd_kwargs=None, **plt_kwargs):
    """Plot a timeseries together with it's associated power spectrum.

    Parameters
    ----------
    sig : 1d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    ts_range : list of [float, float], optional
        The time range to restrict the time series to.
        For visualization only - the power spectrum is computed over the entire time series.
    f_range : list of [float, float], optional
        The frequency range to restrict the power spectrum to.
    spectrum_kwargs : dict, optional
        Keyword arguments for computing the power spectrum.
        See `compute_spectrum` for details.
    start_val : float, optional
        The starting value for the time definition for the time series.
        If not provided, defaults to zero.
    ts_kwargs : dict, optional
        Keyword arguments for customizing the time series plot.
    psd_kwargs : dict, optional
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

    if ts_range:
        ts_kwargs['xlim'] = ts_range
    times = create_times(len(sig) / fs, fs, start_val=start_val)
    plot_time_series(times, sig, ax=ax1, **plt_kwargs,
                     **ts_kwargs if ts_kwargs else {})

    freqs, psd = compute_spectrum(sig, fs, **spectrum_kwargs if spectrum_kwargs else {})
    if f_range:
        freqs, psd = trim_spectrum(freqs, psd, f_range)
    plot_power_spectra(freqs, psd, ax=ax2, **plt_kwargs,
                       **psd_kwargs if psd_kwargs else {})
