"""Analyze the variance of power spectra, using SCV and related metrics."""

import numpy as np
from scipy.signal import spectrogram

from neurodsp.utils import discard_outliers
from neurodsp.spectral.checks import check_spg_settings
from neurodsp.spectral.utils import trim_spectrum

###################################################################################################
###################################################################################################

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
    spect_cv : 1d array
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

    spect_cv = np.std(spg, axis=-1) / np.mean(spg, axis=-1)

    return freqs, spect_cv


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
    t_inds : 1d array
        Array of time indices, for 'rolling' resampling. If 'bootstrap', t_inds = None.
    spect_cv_rs : 1d array
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
        spect_cv_rs = np.zeros((len(freqs), ndraws))

        # Repeated subsampling of spectrogram randomly, with replacement between draws
        for draw in range(ndraws):
            idx = np.random.choice(spg.shape[1], size=nslices, replace=False)
            spect_cv_rs[:, draw] = np.std(
                spg[:, idx], axis=-1) / np.mean(spg[:, idx], axis=-1)

        t_inds = None  # no time component, return nothing

    elif method == 'rolling':

        # Params: number of slices of STFT to compute SCV over & number of slices to roll forward
        #   Defaults to 10 STFT slices, move forward by 5 slices
        if rs_params is None:
            rs_params = (10, 5)

        nslices, nsteps = rs_params
        outlen = int(np.ceil((spg.shape[1] - nslices) / float(nsteps))) + 1
        spect_cv_rs = np.zeros((len(freqs), outlen))
        for ind in range(outlen):
            curblock = spg[:, nsteps * ind:nslices + nsteps * ind]
            spect_cv_rs[:, ind] = np.std(
                curblock, axis=-1) / np.mean(curblock, axis=-1)

        # Grab the time indices from the spectrogram
        t_inds = ts[0::nsteps]

    else:
        raise ValueError('Unknown resampling method: %s' % method)

    return freqs, t_inds, spect_cv_rs


def compute_spectral_hist(sig, fs, window='hann', nperseg=None, noverlap=None,
                          nbins=50, f_lim=(0., 100.), cutpct=(0., 100.)):
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
    f_lim : tuple, optional, default: (0, 100)
        Frequency range of the spectrogram across which to compute the histograms, as (start, end), in Hz.
    cutpct : tuple, optional, default: (0, 100)
        Power percentile at which to draw the lower and upper bin limits, as (low, high).

    Returns
    -------
    freqs : 1d array
        Array of frequencies.
    power_bins : 1d array
        Histogram bins used to compute the distribution.
    spect_hist : 2d array
        Power distribution at every frequency, nbins x fs 2D matrix.

    Notes
    -----
    The histogram bins are the same for every frequency, thus evenly spacing the global min and max power.
    """

    # Compute spectrogram of data
    nperseg, noverlap = check_spg_settings(fs, window, nperseg, noverlap)
    freqs, _, spg = spectrogram(sig, fs, window, nperseg, noverlap, return_onesided=True)

    # Get log10 power & limit to frequency range of interest before binning
    # ToDo / Note: currently includes a hack to maintain test shape
    ps = np.transpose(np.log10(spg))
    freqs, ps = trim_spectrum(freqs, ps, [f_lim[0], f_lim[1] - 1/fs])

    # Prepare bins for power - min and max of bins determined by power cutoff percentage
    power_min, power_max = np.percentile(np.ndarray.flatten(ps), cutpct)
    power_bins = np.linspace(power_min, power_max, nbins + 1)

    # Compute histogram of power for each frequency
    spect_hist = np.zeros((len(ps[0]), nbins))
    for ind in range(len(ps[0])):
        spect_hist[ind], _ = np.histogram(ps[:, ind], power_bins)
        spect_hist[ind] = spect_hist[ind] / sum(spect_hist[ind])

    # Flip output for more sensible plotting direction
    spect_hist = np.transpose(spect_hist)
    spect_hist = np.flipud(spect_hist)

    return freqs, power_bins, spect_hist
