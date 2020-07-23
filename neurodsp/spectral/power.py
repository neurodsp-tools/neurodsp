"""Compute spectral measures that measure spectral power.

Notes
-----
Mostly relies on scipy.signal.spectrogram and numpy.fft
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
"""

import numpy as np
from scipy.signal import spectrogram, medfilt

from neurodsp.utils.core import get_avg_func
from neurodsp.utils.data import create_freqs
from neurodsp.utils.decorators import multidim
from neurodsp.utils.outliers import discard_outliers
from neurodsp.timefrequency.wavelets import compute_wavelet_transform
from neurodsp.spectral.utils import trim_spectrum
from neurodsp.spectral.checks import check_spg_settings

###################################################################################################
###################################################################################################

def compute_spectrum(sig, fs, method='welch', avg_type='mean', **kwargs):
    """Compute the power spectral density of a time series.

    Parameters
    -----------
    sig : 1d or 2d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    method : {'welch', 'wavelet', 'medfilt'}, optional
        Method to use to estimate the power spectrum.
    avg_type : {'mean', 'median'}, optional
        If relevant, the method to average across windows to create the spectrum.
    **kwargs
        Keyword arguments to pass through to the function that calculates the spectrum.

    Returns
    -------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spectrum : 1d or 2d array
        Power spectral density.

    Examples
    --------
    Compute the power spectrum of a simulated time series:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs, spectrum = compute_spectrum(sig, fs=500)
    """

    if method == 'welch':
        return compute_spectrum_welch(sig, fs, avg_type=avg_type, **kwargs)

    elif method == 'wavelet':
        return compute_spectrum_wavelet(sig, fs, avg_type=avg_type, **kwargs)

    elif method == 'medfilt':
        return compute_spectrum_medfilt(sig, fs, **kwargs)

    else:
        raise ValueError('Unknown power spectrum method: %s' % method)


@multidim(select=[0])
def compute_spectrum_wavelet(sig, fs, freqs, avg_type='mean', **kwargs):
    """Compute the power spectral density using wavelets.

    Parameters
    ----------
    sig : 1d or 2d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    freqs : 1d array or list of float
        If array, frequency values to estimate with morlet wavelets.
        If list, define the frequency range, as [freq_start, freq_stop, freq_step].
        The `freq_step` is optional, and defaults to 1. Range is inclusive of `freq_stop` value.
    avg_type : {'mean', 'median'}, optional
        Method to average across the windows.
    **kwargs
        Optional inputs for using wavelets.

    Returns
    -------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spectrum : 1d or 2d array
        Power spectral density.

    Examples
    --------
    Compute the power spectrum of a simulated time series using wavelets:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs, spectrum = compute_spectrum_wavelet(sig, fs=500, freqs=[1, 30])
    """

    if isinstance(freqs, (tuple, list)):
        freqs = create_freqs(*freqs)

    # Compute the wavelet transform
    mwt = compute_wavelet_transform(sig, fs, freqs, **kwargs)

    # Convert the wavelet coefficient outputs to units of power
    mwt_power = abs(mwt)**2

    # Create the power spectrum by averaging across the time dimension
    spectrum = get_avg_func(avg_type)(mwt_power, axis=1)

    return freqs, spectrum


def compute_spectrum_welch(sig, fs, avg_type='mean', window='hann',
                           nperseg=None, noverlap=None,
                           f_range=None, outlier_percent=None):
    """Compute the power spectral density using Welch's method.

    Parameters
    -----------
    sig : 1d or 2d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    avg_type : {'mean', 'median'}, optional
        Method to average across the windows:

        * 'mean' is the same as Welch's method, taking the mean across FFT windows.
        * 'median' uses median across FFT windows instead of the mean, to minimize outlier effects.
    window : str or tuple or array_like, optional, default: 'hann'
        Desired window to use. See scipy.signal.get_window for a list of available windows.
        If array_like, the array will be used as the window and its length must be nperseg.
    nperseg : int, optional
        Length of each segment, in number of samples.
        If None, and window is str or tuple, is set to 1 second of data.
        If None, and window is array_like, is set to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments.
        If None, noverlap = nperseg // 8.
    f_range : list of [float, float], optional
        Frequency range to sub-select from the power spectrum.
    outlier_percent : float, optional
        The percentage of outlier values to be removed. Must be between 0 and 100.

    Returns
    -------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spectrum : 1d or 2d array
        Power spectral density.

    Examples
    --------
    Compute the power spectrum of a simulated time series using Welch's method:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation': {'freq': 10}})
    >>> freqs, spec = compute_spectrum_welch(sig, fs=500)
    """

    # Calculate the short time Fourier transform with signal.spectrogram
    nperseg, noverlap = check_spg_settings(fs, window, nperseg, noverlap)
    freqs, _, spg = spectrogram(sig, fs, window, nperseg, noverlap)

    # Throw out outliers if indicated
    if outlier_percent is not None:
        spg = discard_outliers(spg, outlier_percent)

    # Average across windows
    spectrum = get_avg_func(avg_type)(spg, axis=-1)

    # Trim spectrum, if requested
    if f_range:
        freqs, spectrum = trim_spectrum(freqs, spectrum, f_range)

    return freqs, spectrum


@multidim(select=[0])
def compute_spectrum_medfilt(sig, fs, filt_len=1., f_range=None):
    """Compute the power spectral density as a smoothed FFT.

    Parameters
    ----------
    sig : 1d or 2d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    filt_len : float, optional, default: 1
        Length of the median filter, in Hz.
    f_range : list of [float, float], optional
        Frequency range to sub-select from the power spectrum.

    Returns
    -------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spectrum : 1d or 2d array
        Power spectral density.

    Examples
    --------
    Compute the power spectrum of a simulated time series as a smoothed FFT:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs, spec = compute_spectrum_medfilt(sig, fs=500)
    """

    # Take the positive half of the spectrum, since it's symmetrical
    ft = np.fft.fft(sig)[:int(np.ceil(len(sig) / 2.))]
    freqs = np.fft.fftfreq(len(sig), 1. / fs)[:int(np.ceil(len(sig) / 2.))]

    # Convert median filter length from Hz to samples, and make sure it is odd
    filt_len_samp = int(filt_len / (freqs[1] - freqs[0]))
    if filt_len_samp % 2 == 0:
        filt_len_samp += 1

    spectrum = medfilt(np.abs(ft)**2. / (fs * len(sig)), filt_len_samp)

    if f_range:
        freqs, spectrum = trim_spectrum(freqs, spectrum, f_range)

    return freqs, spectrum
