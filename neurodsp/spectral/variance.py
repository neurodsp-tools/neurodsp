"""Compute spectral measures that measure spectral variance."""

import numpy as np
from scipy.signal import spectrogram

from neurodsp.utils import discard_outliers
from neurodsp.utils.decorators import multidim
from neurodsp.spectral.utils import trim_spectrum
from neurodsp.spectral.checks import check_spg_settings

###################################################################################################
###################################################################################################

@multidim(select=[0])
def compute_scv(sig, fs, window='hann', nperseg=None, noverlap=0, outlier_pct=None):
    """Compute the spectral coefficient of variation (SCV) at each frequency.

    Parameters
    -----------
    sig : 1d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like, optional, default: 'hann'
        Desired window to use. See scipy.signal.get_window for a list of available windows.
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
        Frequencies at which the measure was calculated.
    scv : 1d array
        Spectral coefficient of variation.

    Notes
    -----
    White noise should have a SCV of 1 at all frequencies.

    Examples
    --------
    Compute the spectral coefficient of variation of a simulated time series:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs, scv = compute_scv(sig, fs=500)
    """

    # Compute spectrogram of data
    nperseg, noverlap = check_spg_settings(fs, window, nperseg, noverlap)
    freqs, _, spg = spectrogram(sig, fs, window, nperseg, noverlap)

    if outlier_pct is not None:
        spg = discard_outliers(spg, outlier_pct)

    scv = np.std(spg, axis=-1) / np.mean(spg, axis=-1)

    return freqs, scv


@multidim(select=[0, 1])
def compute_scv_rs(sig, fs, window='hann', nperseg=None, noverlap=0,
                   method='bootstrap', rs_params=None):
    """Compute a resampled version of the spectral coefficient of variation (SCV).

    Parameters
    -----------
    sig : 1d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like, optional, default: 'hann'
        Desired window to use. See scipy.signal.get_window for a list of available windows.
        If array_like, the array will be used as the window and its length must be nperseg.
    nperseg : int, optional
        Length of each segment, in number of samples.
        If None, and window is str or tuple, is set to 1 second of data.
        If None, and window is array_like, is set to the length of the window.
    noverlap : int, optional, default: 0
        Number of points to overlap between segments.
    method : {'bootstrap', 'rolling'}, optional
        Method of resampling:

        * 'bootstrap' randomly samples a subset of the spectrogram repeatedly.
        * 'rolling' takes the rolling window scv.
    rs_params : tuple, (int, int), optional
        Parameters for resampling algorithm, depending on the method used:

        * If 'bootstrap', rs_params = (n_slices, n_draws), defaults to (10% of slices, 100 draws).
        * If 'rolling', rs_params = (n_slices, n_steps), defaults to (10, 5).

        Where:

        * `n_slices` is the number of slices per draw
        * `n_draws` is the number of random draws
        * `n_steps` is the number of slices to step forward.

    Returns
    -------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    t_inds : 1d array or None
        Time indices at which the measure was calculated.
        This is only returned for 'rolling' resampling. If 'bootstrap', t_inds = None.
    scv_rs : 2d array
        Resampled spectral coefficient of variation.

    Notes
    -----
    In the resampled version, instead of a single estimate of mean and standard deviation,
    the spectrogram is resampled.

    Resampling can be done either randomly (method='bootstrap') or in a time-stepped
    manner (method='rolling').

    Examples
    --------
    Compute the resampled spectral coefficient of variation, using the bootstrap method:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs, t_inds, scv_rs = compute_scv_rs(sig, fs=500, method='bootstrap')
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

        # Repeated sub-sampling of spectrogram randomly, with replacement between draws
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


@multidim(select=[0, 1])
def compute_spectral_hist(sig, fs, window='hann', nperseg=None, noverlap=None,
                          nbins=50, f_range=[0., 100.], cut_pct=[0., 100.]):
    """Compute the distribution of log10 power at each frequency from the signal spectrogram.

    Parameters
    -----------
    sig : 1d array
        Time series of measurement values.
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like, optional, default: 'hann'
        Desired window to use. See scipy.signal.get_window for a list of available windows.
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
        Frequencies at which the measure was calculated.
    power_bins : 1d array
        Histogram bins used to compute the distribution.
    spectral_hist : 2d array
        Power distribution at every frequency, as [n_bins, freqs].

    Notes
    -----
    Histogram bins are the same for every frequency, evenly spacing the global min & max power.

    Examples
    --------
    Compute the distribution of power, which is the spectral histogram:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation': {'freq': 10}})
    >>> freqs, power_bins, spectral_hist = compute_spectral_hist(sig, fs=500)
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
