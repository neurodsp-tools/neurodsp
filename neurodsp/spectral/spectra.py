"""Compute power spectra."""

import numpy as np
from scipy import signal

from neurodsp.utils import discard_outliers
from neurodsp.spectral.utils import trim_spectrum
from neurodsp.spectral.checks import check_spg_settings

###################################################################################################
###################################################################################################

def compute_spectrum(sig, fs, method='mean', window='hann', nperseg=None,
                     noverlap=None, filt_len=1., f_lim=None, spg_outlier_pct=0.):
    """Estimate the power spectral density (PSD) of a time series.

    Parameters
    -----------
    sig : 1d or 2d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    method : {'mean', 'median', 'medfilt'}, optional
        Method to calculate the spectrum:

        * 'mean' is the same as Welch's method (mean of STFT).
        * 'median' uses median of STFT instead of mean to minimize outlier effect.
        * 'medfilt' filters the entire signals raw FFT with a median filter to smooth.
    window : str or tuple or array_like, optional, default: 'hann'
        Desired window to use. Defaults to a Hann window.
        See scipy.signal.get_window for a list of windows and required parameters.
        If array_like, the array will be used as the window and its length must be nperseg.
    nperseg : int, optional
        Length of each segment, in number of samples.
        If None, and window is str or tuple, is set to 1 second of data.
        If None, and window is array_like, is set to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments.
        If None, noverlap = nperseg // 8.
    filt_len : float, optional
        Length of the median filter, in Hz.
        Only used with the 'medfilt' method.
    f_lim : float, optional, default: 1.
        Maximum frequency to keep, in Hz.
        If None, keeps up to Nyquist.
    spg_outlier_pct : float, optional, default: 0.
        Percentage of spectrogram windows with the highest powers to discard prior to averaging.
        Useful for quickly eliminating potential outliers to compute spectrum.
        Must be between 0 and 100.

    Returns
    -------
    freqs : 1d array
        Array of sample frequencies.
    spectrum : 1d or 2d array
        Power spectral density.

    References
    ----------
    Mostly relies on scipy.signal.spectrogram and numpy.fft
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    """

    if method not in ('mean', 'median', 'medfilt'):
        raise ValueError('Unknown power spectrum method: %s' % method)

    if method in ('mean', 'median'):
        return compute_spectrum_welch(sig, fs, method, window,
                                      nperseg, noverlap, f_lim, spg_outlier_pct)

    elif method == 'medfilt':
        return compute_spectrum_medfilt(sig, fs, filt_len, f_lim)


def compute_spectrum_welch(sig, fs, method='mean', window='hann', nperseg=None,
                           noverlap=None, f_lim=None, spg_outlier_pct=0.):
    """Estimate the power spectral density using Welch's method.

    Parameters
    -----------
    sig : 1d or 2d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    method : {'mean', 'median'}, optional
        Method to average across the windows:

        * 'mean' is the same as Welch's method (mean of STFT).
        * 'median' uses median of STFT instead of mean to minimize outlier effect.
    window : str or tuple or array_like, optional, default: 'hann'
        Desired window to use. Defaults to a Hann window.
        See scipy.signal.get_window for a list of windows and required parameters.
        If array_like, the array will be used as the window and its length must be nperseg.
    nperseg : int, optional
        Length of each segment, in number of samples.
        If None, and window is str or tuple, is set to 1 second of data.
        If None, and window is array_like, is set to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments.
        If None, noverlap = nperseg // 8.
    f_lim : float, optional
        Maximum frequency to keep, in Hz.
        If None, keeps up to Nyquist.
    spg_outlier_pct : float, optional, default: 0.
        Percentage of spectrogram windows with the highest powers to discard prior to averaging.
        Useful for quickly eliminating potential outliers to compute spectrum.
        Must be between 0 and 100.

    Returns
    -------
    freqs : 1d array
        Array of sample frequencies.
    spectrum : 1d or 2d array
        Power spectral density.
    """

    # Calculate the short time fourier transform with signal.spectrogram
    nperseg, noverlap = check_spg_settings(fs, window, nperseg, noverlap)
    freqs, _, spg = signal.spectrogram(sig, fs, window, nperseg, noverlap)

    # Pad data to 2D
    if len(sig.shape) == 1:
        sig = sig[np.newaxis :]

    # Throw out outliers if indicated
    if spg_outlier_pct > 0.:
        spg = discard_outliers(spg, spg_outlier_pct)

    if method == 'mean':
        spectrum = np.mean(spg, axis=-1)
    elif method == 'median':
        spectrum = np.median(spg, axis=-1)

    if f_lim:
        freqs, spectrum = trim_spectrum(freqs, spectrum, [freqs[0], f_lim])

    return freqs, spectrum


def compute_spectrum_medfilt(sig, fs, filt_len=1., f_lim=None):
    """Estimate the power spectral densitry as a smoothed FFT.

    Parameters
    ----------
    sig : 1d or 2d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    filt_len : float, optional, default: 1.
        Length of the median filter, in Hz.
    f_lim : float, optional
        Maximum frequency to keep, in Hz. If None, keeps up to Nyquist.

    Returns
    -------
    freqs : 1d array
        Array of sample frequencies.
    spectrum : 1d or 2d array
        Power spectral density.
    """

    # Take the positive half of the spectrum since it's symmetrical
    ft = np.fft.fft(sig)[:int(np.ceil(len(sig) / 2.))]
    freqs = np.fft.fftfreq(len(sig), 1. / fs)[:int(np.ceil(len(sig) / 2.))]  # get freq axis

    # Convert median filter length from Hz to samples
    filt_len_samp = int(int(filt_len / (freqs[1] - freqs[0])) / 2 * 2 + 1)
    spectrum = signal.medfilt(np.abs(ft)**2. / (fs * len(sig)), filt_len_samp)

    if f_lim:
        freqs, spectrum = trim_spectrum(freqs, spectrum, [freqs[0], f_lim])

    return freqs, spectrum
