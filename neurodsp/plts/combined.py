"""Plotting functions for plots with combined panels."""

import matplotlib.pyplot as plt

from neurodsp.utils.data import create_times
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series, plot_multi_time_series
from neurodsp.plts.utils import savefig

###################################################################################################
###################################################################################################

@savefig
def plot_timeseries_and_spectrum(sig, fs, ts_range=None, f_range=None, times=None, start_val=0.,
                                 spectrum_kwargs=None, ts_kwargs=None, psd_kwargs=None,
                                 **plt_kwargs):
    """Plot a timeseries together with it's associated power spectrum.

    Parameters
    ----------
    sig : 1d or 2d array
        Time series to plot.
    fs : float
        Sampling rate, in Hz.
    ts_range : list of [float, float], optional
        The time range to restrict the time series to.
        For visualization only - the power spectrum is computed over the entire time series.
    f_range : list of [float, float], optional
        The frequency range to restrict the power spectrum to.
    times : 1d, optional
        Time definition for the time series to be plotted.
        If not provided, times are recomputed from signal length and sampling rate.
    start_val : float, optional, default: 0.
        The starting value for the time definition for the time series.
        Only used if `times` is not provided.
    spectrum_kwargs : dict, optional
        Keyword arguments for computing the power spectrum.
        See `compute_spectrum` for details.
    ts_kwargs : dict, optional
        Keyword arguments for customizing the time series plot.
    psd_kwargs : dict, optional
        Keyword arguments for customizing the power spectrum plot.
    **plt_kwargs
        Keyword arguments for customizing the plots.
        These arguments are passed to both plot axes.
    """

    # Import spectal functions locally to avoid circular imports
    from neurodsp.spectral import compute_spectrum
    from neurodsp.spectral.utils import trim_spectrum

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

    if not times:
        times = create_times(sig.shape[-1] / fs, fs, start_val=start_val)
    if ts_range:
        ts_kwargs['xlim'] = ts_range

    if sig.ndim == 1:
        plot_time_series(times, sig, ax=ax1, **plt_kwargs,
                         **ts_kwargs if ts_kwargs else {})
    elif sig.ndim == 2:
        plot_multi_time_series(times, sig, ax=ax1, **plt_kwargs,
                               **ts_kwargs if ts_kwargs else {})
    else:
        raise ValueError('Only 1d or 2d inputs are supported.')

    freqs, psd = compute_spectrum(sig, fs, **spectrum_kwargs if spectrum_kwargs else {})
    if f_range:
        freqs, psd = trim_spectrum(freqs, psd, f_range)
    plot_power_spectra(freqs, psd, ax=ax2, **plt_kwargs,
                       **psd_kwargs if psd_kwargs else {})
