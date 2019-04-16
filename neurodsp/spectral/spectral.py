"""Compute spectral measures, including power spectra and spectral variance.

Notes
-----
Mostly relies on scipy.signal.spectrogram and numpy.fft
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
"""

import numpy as np
from scipy.signal import spectrogram, medfilt

from neurodsp.utils import discard_outliers
from neurodsp.utils.core import get_avg_func
from neurodsp.utils.decorators import multidim
from neurodsp.timefrequency.wavelets import morlet_transform
from neurodsp.spectral.utils import trim_spectrum
from neurodsp.spectral.checks import check_spg_settings

###################################################################################################
###################################################################################################

@multidim
def compute_spectrum(sig, fs, method='welch', avg_type='mean', **kwargs):
    """Estimate the power spectral density (PSD) of a time series.

    Parameters
    -----------
    sig : 1d or 2d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    method : {'welch', 'wavelet', 'medfilt'}
        Method to use to estimate the power spectrum.
    avg_type : {'mean', 'median'}, optional
        If relevant, the method to average across windows to create the spectrum.
    **kwargs
        Keyword arguments to pass through to function that calculates the spectrum.

    Returns
    -------
    freqs : 1d array
        Array of sample frequencies.
    spectrum : 1d or 2d array
        Power spectral density.
    """

    if method not in ('welch', 'medfilt', 'wavelet'):
        raise ValueError('Unknown power spectrum method: %s' % method)

    if method == 'welch':
        return compute_spectrum_welch(sig, fs, avg_type=avg_type, **kwargs)

    elif method == 'wavelet':
        return compute_spectrum_wavelet(sig, fs, avg_type=avg_type, **kwargs)

    elif method == 'medfilt':
        return compute_spectrum_medfilt(sig, fs, **kwargs)


@multidim
def compute_spectrum_wavelet(sig, fs, freqs, avg_type='mean', **kwargs):
    """Estimate the power spectral densitry using wavelets.

    Parameters
    ----------
    sig : 1d or 2d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    avg_type : {'mean', 'median'}, optional
        Method to average across the windows.
    **kwargs
        Optional inputs for using wavelets.

    Returns
    -------
    freqs : 1d array
        Array of sample frequencies.
    spectrum : 1d or 2d array
        Power spectral density.
    """

    mwt = morlet_transform(sig, fs, freqs, **kwargs)
    spectrum = get_avg_func(avg_type)(mwt, axis=0)

    return freqs, spectrum


@multidim
def compute_spectrum_welch(sig, fs, avg_type='mean', window='hann',
                           nperseg=None, noverlap=None,
                           f_range=None, outlier_pct=None):
    """Estimate the power spectral density using Welch's method.

    Parameters
    -----------
    sig : 1d or 2d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    avg_type : {'mean', 'median'}, optional
        Method to average across the windows:

        * 'mean' is the same as Welch's method, taking the mean across FFT windows.
        * 'median' uses median across FFT windows instead of the mean, to minimize outlier effect.
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
    f_range : list of [float, float] optional
        Frequency range to sub-select from the power spectrum.
    outlier_pct : float, optional
        Percentage of the windows with the lowest and highest total log power to discard.
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
    freqs, _, spg = spectrogram(sig, fs, window, nperseg, noverlap)

    # Pad data to 2D
    if len(sig.shape) == 1:
        sig = sig[np.newaxis :]

    # Throw out outliers if indicated
    if outlier_pct is not None:
        spg = discard_outliers(spg, outlier_pct)

    # Average across windows
    spectrum = get_avg_func(avg_type)(spg, axis=-1)

    # Trim spectrum, if requested
    if f_range:
        freqs, spectrum = trim_spectrum(freqs, spectrum, f_range)

    return freqs, spectrum


@multidim
def compute_spectrum_medfilt(sig, fs, filt_len=1., f_range=None):
    """Estimate the power spectral densitry as a smoothed FFT.

    Parameters
    ----------
    sig : 1d or 2d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    filt_len : float, optional, default: 1.
        Length of the median filter, in Hz.
    f_range : list of [float, float] optional
        Frequency range to sub-select from the power spectrum.

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
    spectrum = medfilt(np.abs(ft)**2. / (fs * len(sig)), filt_len_samp)

    if f_range:
        freqs, spectrum = trim_spectrum(freqs, spectrum, f_range)

    return freqs, spectrum


@multidim
def compute_scv(sig, fs, window='hann', nperseg=None, noverlap=0, outlier_pct=None):
    """Compute the spectral coefficient of variation (SCV) at each frequency.

    Parameters
    -----------
    sig : 1d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like, optional, default='hann'
        Desired window to use. Defaults to a Hann window.
        See scipy.signal.get_window for a list of windows and required parameters.
        If array_like, the array will be used as the window and its length must be nperseg.
    nperseg : int, optional
        Length of each segment, in number of samples.
        If None, and window is str or tuple, is set to 1 second of data.
        If None, and window is array_like, is set to the length of the window.
    noverlap : int, optional, default: 0
        Number of points to overlap between segments.
    outlier_pct : float, optional
        Percentage of the windows with the lowest and highest total log power to discard.
        Must be between 0 and 100.

    Returns
    -------
    freqs : 1d array
        Array of sample frequencies.
    scv : 1d array
        Spectral coefficient of variation.

    Notes
    -----
    White noise should have a SCV of 1 at all frequencies.
    """

    # Compute spectrogram of data
    nperseg, noverlap = check_spg_settings(fs, window, nperseg, noverlap)
    freqs, _, spg = spectrogram(sig, fs, window, nperseg, noverlap)

    if outlier_pct is not None:
        spg = discard_outliers(spg, outlier_pct)

    scv = np.std(spg, axis=-1) / np.mean(spg, axis=-1)

    return freqs, scv


@multidim
def compute_scv_rs(sig, fs, window='hann', nperseg=None, noverlap=0,
                   method='bootstrap', rs_params=None):
    """Resampled version of scv: instead of a single estimate of mean and standard deviation,
    the spectrogram is resampled, either randomly (bootstrap) or time-stepped (rolling).

    Parameters
    -----------
    sig : 1d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like, optional, default='hann'
        Desired window to use. Defaults to a Hann window.
        See scipy.signal.get_window for a list of windows and required parameters.
        If array_like, the array will be used as the window and its length must be nperseg.
    nperseg : int, optional
        Length of each segment, in number of samples.
        If None, and window is str or tuple, is set to 1 second of data.
        If None, and window is array_like, is set to the length of the window.
    noverlap : int, optional, default: 0
        Number of points to overlap between segments.
    method : {'bootstrap', 'rolling'}, optional
        Method of resampling.
        'bootstrap' randomly samples a subset of the spectrogram repeatedly.
        'rolling' takes the rolling window scv.
    rs_params : tuple, (int, int), optional
        Parameters for resampling algorithm, depending on the method used.
        If 'bootstrap', rs_params = (nslices, ndraws), defaults to (10% of slices, 100 draws).
        This specifies the number of slices per draw, and number of random draws.
        If 'rolling', rs_params = (nslices, nsteps), defaults to (10, 5).
        This specifies the number of slices per draw, and number of slices to step forward.

    Returns
    -------
    freqs : 1d array
        Array of sample frequencies.
    t_inds : 1d array or None
        Array of time indices, for 'rolling' resampling. If 'bootstrap', t_inds = None.
    scv_rs : 2d array
        Resampled spectral coefficient of variation.
    """

    # Compute spectrogram of data
    nperseg, noverlap = check_spg_settings(fs, window, nperseg, noverlap)
    freqs, ts, spg = spectrogram(sig, fs, window, nperseg, noverlap)

    if method == 'bootstrap':

        # Params: number of slices of STFT to compute SCV over & number of draws
        #   Defaults to draw 1/10 of STFT slices, 100 draws
        if rs_params is None:
            rs_params = (int(spg.shape[1] / 10.), 100)

        nslices, ndraws = rs_params
        scv_rs = np.zeros((len(freqs), ndraws))

        # Repeated subsampling of spectrogram randomly, with replacement between draws
        for draw in range(ndraws):
            idx = np.random.choice(spg.shape[1], size=nslices, replace=False)
            scv_rs[:, draw] = np.std(
                spg[:, idx], axis=-1) / np.mean(spg[:, idx], axis=-1)

        t_inds = None  # no time component, return nothing

    elif method == 'rolling':

        # Params: number of slices of STFT to compute SCV over & number of slices to roll forward
        #   Defaults to 10 STFT slices, move forward by 5 slices
        if rs_params is None:
            rs_params = (10, 5)

        nslices, nsteps = rs_params
        outlen = int(np.ceil((spg.shape[1] - nslices) / float(nsteps))) + 1
        scv_rs = np.zeros((len(freqs), outlen))
        for ind in range(outlen):
            curblock = spg[:, nsteps * ind:nslices + nsteps * ind]
            scv_rs[:, ind] = np.std(
                curblock, axis=-1) / np.mean(curblock, axis=-1)

        # Grab the time indices from the spectrogram
        t_inds = ts[0::nsteps]

    else:
        raise ValueError('Unknown resampling method: %s' % method)

    return freqs, t_inds, scv_rs


@multidim
def compute_spectral_hist(sig, fs, window='hann', nperseg=None, noverlap=None,
                          nbins=50, f_range=[0., 100.], cut_pct=[0., 100.]):
    """Compute the distribution of log10 power at each frequency from the signal spectrogram.

    Parameters
    -----------
    sig : 1d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like, optional, default='hann'
        Desired window to use. Defaults to a Hann window.
        See scipy.signal.get_window for a list of windows and required parameters.
        If array_like, the array will be used as the window and its length must be nperseg.
    nperseg : int, optional
        Length of each segment, in number of samples.
        If None, and window is str or tuple, is set to 1 second of data.
        If None, and window is array_like, is set to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If None, noverlap = nperseg // 2.
    nbins : int, optional, default: 50
        Number of histogram bins to use.
    f_range : list of [float, float], optional, default: [0, 100]
        Frequency range of the spectrogram to compute the histograms, as [start, end], in Hz.
    cut_pct : list of [float, float], optional, default: [0, 100]
        Power percentile at which to draw the lower and upper bin limits, as [low, high], in Hz.

    Returns
    -------
    freqs : 1d array
        Array of frequencies.
    power_bins : 1d array
        Histogram bins used to compute the distribution.
    spectral_hist : 2d array
        Power distribution at every frequency, nbins x fs 2D matrix.

    Notes
    -----
    The histogram bins are the same for every frequency, thus evenly spacing the global min and max power.
    """

    # Compute spectrogram of data
    nperseg, noverlap = check_spg_settings(fs, window, nperseg, noverlap)
    freqs, _, spg = spectrogram(sig, fs, window, nperseg, noverlap, return_onesided=True)

    # Get log10 power & limit to frequency range of interest before binning
    ps = np.transpose(np.log10(spg))
    freqs, ps = trim_spectrum(freqs, ps, f_range)

    # Prepare bins for power - min and max of bins determined by power cutoff percentage
    power_min, power_max = np.percentile(np.ndarray.flatten(ps), cut_pct)
    power_bins = np.linspace(power_min, power_max, nbins + 1)

    # Compute histogram of power for each frequency
    spectral_hist = np.zeros((len(ps[0]), nbins))
    for ind in range(len(ps[0])):
        spectral_hist[ind], _ = np.histogram(ps[:, ind], power_bins)
        spectral_hist[ind] = spectral_hist[ind] / sum(spectral_hist[ind])

    # Flip output for more sensible plotting direction
    spectral_hist = np.transpose(spectral_hist)
    spectral_hist = np.flipud(spectral_hist)

    return freqs, power_bins, spectral_hist
