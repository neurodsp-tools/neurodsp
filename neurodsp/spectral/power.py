"""Compute spectral power.

Notes
-----
Mostly relies on scipy.signal.spectrogram and numpy.fft
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
"""

import numpy as np
from scipy.signal import spectrogram, medfilt
from scipy.fft import next_fast_len

from neurodsp.utils.core import get_avg_func
from neurodsp.utils.data import create_freqs
from neurodsp.utils.decorators import multidim
from neurodsp.utils.checks import check_param_options
from neurodsp.utils.outliers import discard_outliers
from neurodsp.timefrequency.wavelets import compute_wavelet_transform
from neurodsp.spectral.utils import trim_spectrum, window_pad
from neurodsp.spectral.checks import check_spg_settings, check_mt_settings

###################################################################################################
###################################################################################################

def compute_spectrum(sig, fs, method='welch', **kwargs):
    """Compute the power spectral density of a time series.

    Parameters
    ----------
    sig : 1d or 2d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    method : {'welch', 'wavelet', 'medfilt', 'multitaper'}, optional
        Method to use to estimate the power spectrum.
    **kwargs
        Keyword arguments to pass through to the function that calculates the spectrum.
        See `compute_spectrum_{welch, wavelet, medfilt}` for details.

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

    check_param_options(method, 'method', ['welch', 'wavelet', 'medfilt', 'multitaper'])
    _spectrum_input_checks(method, kwargs)

    if method == 'welch':
        return compute_spectrum_welch(sig, fs, **kwargs)

    elif method == 'wavelet':
        return compute_spectrum_wavelet(sig, fs, **kwargs)

    elif method == 'medfilt':
        return compute_spectrum_medfilt(sig, fs, **kwargs)

    elif method == 'multitaper':
        return compute_spectrum_multitaper(sig, fs, **kwargs)


SPECTRUM_INPUTS = {
    'welch' : ['avg_type', 'window', 'nperseg', 'noverlap', 'nfft', \
               'fast_len', 'f_range', 'outlier_percent'],
    'wavelet' : ['freqs', 'avg_type', 'n_cycles', 'scaling', 'norm'],
    'medfilt' : ['filt_len', 'f_range'],
}


def _spectrum_input_checks(method, kwargs):
    """Check inputs to `compute_spectrum` match spectral estimation method."""

    for param in kwargs.keys():
        assert param in SPECTRUM_INPUTS[method], \
            'Parameter {} not expected for {} estimation method'.format(param, method)


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
                           nperseg=None, noverlap=None, nfft=None,
                           fast_len=False, f_range=None, outlier_percent=None):
    """Compute the power spectral density using Welch's method.

    Parameters
    ----------
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
    nfft : int, optional
        Number of samples per window. Requires nfft > nperseg.
        Windows are zero-padded by the difference, nfft - nperseg.
    fast_len : bool, optional, default: False
        Moves nperseg to the fastest length to reduce computation.
        See scipy.fft.next_fast_len for details.
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

    Notes
    -----
    - Welch's method ([1]_) computes a power spectra by averaging over windowed FFTs.

    References
    ----------
    .. [1] Welch, P. (1967). The use of fast Fourier transform for the estimation of power
           spectra: A method based on time averaging over short, modified periodograms.
           IEEE Transactions on Audio and Electroacoustics, 15(2), 70–73.
           DOI: https://doi.org/10.1109/TAU.1967.1161901

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

    # Pad signal if requested
    if nfft is not None and nfft < nperseg:
        raise ValueError('nfft must be greater than nperseg.')
    elif nfft is not None:
        npad = nfft - nperseg
        noverlap = nperseg // 8 if noverlap is None else noverlap
        sig, nperseg, noverlap = window_pad(sig, nperseg, noverlap, npad, fast_len)
    elif fast_len:
        nperseg = next_fast_len(nperseg)

    # Compute spectrogram
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


def compute_spectrum_multitaper(sig, fs, bandwidth=None, n_tapers=None,
                                low_bias=True, eigenvalue_weighting=True):
    """Compute the power spectral density using the multi-taper method.

    Parameters
    ----------
    sig : 1d or 2d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    bandwidth : float, optional
        Frequency bandwidth of multi-taper window function.
        If not provided, defaults to 8 * fs / n_samples.
    n_tapers : int, optional
        Number of slepian windows used to compute the spectrum.
        If not provided, defaults to bandwidth * n_samples / fs.
    low_bias : bool, optional, default: True
        If True, only use tapers with concentration ratio > 0.9.
    eigenvalue_weighting : bool, optional
        If True, weight spectral estimates by the concentration ratio of
        their respective tapers before combining. Default is True.

    Returns
    -------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spectrum : 1d or 2d array
        Power spectral density using multi-taper method.

    Examples
    --------
    Compute the power spectrum of a simulated time series using the multitaper method:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs, spec = compute_spectrum_multitaper(sig, fs=500)
    """

    from scipy.signal.windows import dpss

    # Compute signal length based on input shape
    sig_len = sig.shape[sig.ndim - 1]

    # check settings
    nw, n_tapers = check_mt_settings(sig_len, fs, bandwidth, n_tapers)

    # Create slepian sequences
    slepian_sequences, ratios = dpss(sig_len, nw, n_tapers, return_ratios=True)

    # Drop tapers with low concentration
    if low_bias:
        slepian_sequences = slepian_sequences[ratios > 0.9]
        ratios = ratios[ratios > 0.9]
        if len(slepian_sequences) == 0:
            raise ValueError("No tapers with concentration ratio > 0.9. "
                             "Could not compute spectrum with low_bias=True.")

    # Compute Fourier transform on signal weighted by each slepian sequence
    freqs = np.fft.rfftfreq(sig_len, 1. /fs)
    spectra = np.abs(np.fft.rfft(slepian_sequences[:, np.newaxis] * sig)) ** 2

    # combine estimates to compute final spectrum
    if eigenvalue_weighting:
        # weight estimates by concentration ratios and combine
        spectra_weighted = spectra * ratios[:, np.newaxis, np.newaxis]
        spectrum = np.sum(spectra_weighted, axis=0) / np.sum(ratios)

    else:
        # Average spectral estimates
        spectrum = spectra.mean(axis=0)

    # Convert output to 1d if necessary
    if sig.ndim == 1:
        spectrum = spectrum[0]

    return freqs, spectrum
