"""
spectral.py
Frequency domain analysis of neural signals: creating PSD, fitting 1/f, spectral histograms
"""

import numpy as np
from scipy import signal
import matplotlib.pylab as plt
from sklearn import linear_model


def psd(x, Fs, method='mean', window='hann', nperseg=None, noverlap=None, filtlen=1.):
    """
    Estimating the power spectral density (PSD) of a time series from short-time Fourier
    Transform (mean, median), or the entire signal's FFT smoothed (medfilt)

    Parameters
    -----------
    x : array_like 1d
            Time series of measurement values
    Fs : float, Hz
            Sampling frequency of the x time series.
    method : { 'mean', 'median', 'medfilt'}, optional
            Methods to calculate the PSD. 'mean' is the same as Welch's method (mean of STFT), 'median' uses median of STFT instead of mean to minimize outlier effect. 'medfilt' filters the entire signals raw FFT with a median filter to smooth. Defaults to 'mean'.
    The next 3 parameters are only relevant for method = {'mean', 'median'}
    window : str or tuple or array_like, optional
            Desired window to use. See scipy.signal.get_window for a list of windows and required parameters. If window is array_like it will be used directly as the window and its length must be nperseg. Defaults to a Hann window.
    nperseg : int, optional
            Length of each segment. Defaults to None, but if window is str or tuple, is set to 1 second of data, and if window is array_like, is set to the length of the window.
    noverlap : int, optional
            Number of points to overlap between segments. If None, noverlap = nperseg // 2. Defaults to None.
    filten : float, Hz, optional
                        (For medfilt method) Length of median filter in Hz.

    Returns
    -------
    freq : ndarray
            Array of sample frequencies.
    Pxx : ndarray
            Power spectral density of x.

    References
    ----------
    Mostly relies on scipy.signal.spectrogram and numpy.fft
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    """

    if method in ('mean', 'median'):
        # welch-style spectrum (mean/median of STFT)
        if nperseg is None:
            if type(window) in (str, tuple):
                # window is a string or tuple, defaults to 1 second of data
                nperseg = int(Fs)
            else:
                # window is an array, defaults to window length
                nperseg = len(window)
        else:
            nperseg = int(nperseg)

        if noverlap is not None:
            noverlap = int(noverlap)

        # call signal.spectrogram function in scipy to compute STFT
        freq, t, spg = signal.spectrogram(x, Fs, window, nperseg, noverlap)
        if method is 'mean':
            Pxx = np.mean(spg, axis=-1)
        elif method is 'median':
            Pxx = np.median(spg, axis=-1)

    elif method is 'medfilt':
        # median filtered FFT spectrum
        # take the positive half of the spectrum since it's symmetrical
        FT = np.fft.fft(x)[:int(np.ceil(len(x) / 2.))]
        freq = np.fft.fftfreq(len(x), 1. / Fs)[:int(np.ceil(len(x) / 2.))]  # get freq axis
        # convert median filter length from Hz to samples
        filtlen_samp = int(int(filtlen / (freq[1] - freq[0])) / 2 * 2 + 1)
        Pxx = signal.medfilt(np.abs(FT)**2. / (Fs * len(x)), filtlen_samp)

    else:
        raise ValueError('Unknown PSD method: %s' % method)

    return freq, Pxx


def scv(x, Fs, window='hann', nperseg=None, noverlap=0, outlierpct=None):
    """
    Compute the spectral coefficient of variation (SCV) at each frequency.
    White noise should have a SCV of 1 at all frequencies.

    Parameters
    -----------
    x : array_like 1d
            Time series of measurement values
    Fs : float, Hz
            Sampling frequency of the x time series.
    window : str or tuple or array_like, optional
            Desired window to use. See scipy.signal.get_window for a list of windows and required parameters. If window is array_like it will be used directly as the window and its length must be nperseg. Defaults to a Hann window.
    nperseg : int, optional
            Length of each segment. Defaults to None, but if window is str or tuple, is set to 1 second of data, and if window is array_like, is set to the length of the window.
    noverlap : int, optional
            Number of points to overlap between segments. Defaults to 0 for independence.
    outlierpct : float, percent, optional
            Discarding a percentage of the windows with the lowest and highest total log power.

    Returns
    -------
    freq : ndarray
            Array of sample frequencies.
    SCV : ndarray
            Spectral coefficient of variation.
    """
    if nperseg is None:
        if type(window) in (str, tuple):
            # window is a string or tuple, defaults to 1 second of data
            nperseg = int(Fs)
        else:
            # window is an array, defaults to window length
            nperseg = len(window)
    else:
        nperseg = int(nperseg)

    if noverlap is not None:
        noverlap = int(noverlap)

    freq, t, spg = signal.spectrogram(x, Fs, window, nperseg, noverlap)
    if outlierpct is not None:
        # discard time windows with very low and very high powers
        discard = int(spg.shape[1] / 100. * outlierpct)
        outlieridx = np.argsort(np.mean(np.log10(spg), axis=0))[discard:-discard]
        spg = spg[:, outlieridx]

    spectcv = np.std(spg, axis=-1) / np.mean(spg, axis=-1)
    return freq, spectcv


def scv_rs(x, Fs, window='hann', nperseg=None, noverlap=0, method='bootstrap', rs_params=None):
    """
    Resampled version of scv: instead of a single estimate of mean and standard deviation, the spectrogram is resampled, either randomly (bootstrap) or time-stepped (rolling).

    Parameters
    -----------
    x : array_like 1d
            Time series of measurement values
    Fs : float, Hz
            Sampling frequency of the x time series.
    window : str or tuple or array_like, optional
            Desired window to use. See scipy.signal.get_window for a list of windows and required parameters. If window is array_like it will be used directly as the window and its length must be nperseg. Defaults to a Hann window.
    nperseg : int, optional
            Length of each segment. Defaults to None, but if window is str or tuple, is set to 1 second of data, and if window is array_like, is set to the length of the window.
    noverlap : int, optional
            Number of points to overlap between segments. Defaults to 0 for independence.
    method : {'bootstrap', 'rolling'}, optional
            Method of resampling: bootstrap randomly samples a subset of the spectrogram repeatedly, while rolling takes the rolling window scv.
            Defaults to 'bootstrap'.
    rs_params : tuple, (int, int), optional
            Parameters for resampling algorithm.
            If 'bootstrap', rs_params = (nslices, ndraws), representing number of slices per draw, and number of random draws, defaults to (10% of slices, 100 draws).
            If 'rolling', rs_params = (nslices, nsteps), representing number of slices per draw, and number of slices to step forward, defaults to (10, 5).
    Returns
    -------
    freq : ndarray
            Array of sample frequencies.
    T : ndarray
            Array of time indices, for 'rolling' resampling. If 'bootstrap', T = None.
    spectcv_rs : ndarray
            Resampled spectral coefficient of variation.
    """
    if nperseg is None:
        if type(window) in (str, tuple):
            # window is a string or tuple, defaults to 1 second of data
            nperseg = int(Fs)
        else:
            # window is an array, defaults to window length
            nperseg = len(window)
    else:
        nperseg = int(nperseg)

    if noverlap is not None:
        noverlap = int(noverlap)
    # compute spectrogram
    freq, t, spg = signal.spectrogram(x, Fs, window, nperseg, noverlap)

    if method is 'bootstrap':
        # params are number of slices of STFT to compute SCV over, and number of draws
        if rs_params is None:
            # defaults to draw 1/10 of STFT slices, 100 draws
            rs_params = (int(spg.shape[1] / 10.), 100)

        nslices, ndraws = rs_params
        spectcv_rs = np.zeros((len(freq), ndraws))
        for draw in range(ndraws):
            # repeated subsampling of spectrogram randomly, with replacement between draws
            idx = np.random.choice(spg.shape[1], size=nslices, replace=False)
            spectcv_rs[:, draw] = np.std(spg[:, idx], axis=-1) / np.mean(spg[:, idx], axis=-1)

        T = None  # no time component, return nothing

    elif method is 'rolling':
        # params are number of slices of STFT to compute SCV over, and number of slices to roll forward
        if rs_params is None:
            # defaults to 10 STFT slices, move forward by 5 slices
            rs_params = (10, 5)

        nslices, nsteps = rs_params
        outlen = int(np.ceil((spg.shape[1] - nslices) / float(nsteps))) + 1
        spectcv_rs = np.zeros((len(freq), outlen))
        for ind in range(outlen):
            curblock = spg[:, nsteps * ind:nslices + nsteps * ind]
            spectcv_rs[:, ind] = np.std(curblock, axis=-1) / np.mean(curblock, axis=-1)

        T = t[0::nsteps]  # grab the time indices from the spectrogram

    else:
        raise ValueError('Unknown resampling method: %s' % method)

    return freq, T, spectcv_rs


def spectral_hist(x, Fs, window='hann', nperseg=None, noverlap=None, nbins=50, flim=(0., 100.), cutpct=(0., 100.)):
    """
    Compute the distribution of log10 power at each frequency from the signal spectrogram.
        The histogram bins are the same for every frequency, thus evenly spacing the global min and max power

    Parameters
    -----------
    x : array_like 1d
            Time series of measurement values
    Fs : float, Hz
            Sampling frequency of the x time series.
    window : str or tuple or array_like, optional
            Desired window to use. See scipy.signal.get_window for a list of windows and required parameters. If window is array_like it will be used directly as the window and its length must be nperseg. Defaults to a Hann window.
    nperseg : int, optional
            Length of each segment. Defaults to None, but if window is str or tuple, is set to 1 second of data, and if window is array_like, is set to the length of the window.
    noverlap : int, optional
            Number of points to overlap between segments. If None, noverlap = nperseg // 2. Defaults to None.
    nbins : int, optional
            Number of histogram bins to use, defaults to 50
    flim : tuple, (start, end) in Hz, optional
            Frequency range of the spectrogram across which to compute the histograms. Default to (0., 100.)
    cutpct : tuple, (low, high), in percentage, optional
                Power percentile at which to draw the lower and upper bin limits. Default to (0., 100.)

    Returns
    -------
    freq : ndarray
            Array of frequencies.
    power_bins : ndarray
            Histogram bins used to compute the distribution.
    spect_hist : ndarray (2D)
            Power distribution at every frequency, nbins x freqs 2D matrix
    """

    if nperseg is None:
        if type(window) in (str, tuple):
            # window is a string or tuple, defaults to 1 second of data
            nperseg = int(Fs)
        else:
            # window is an array, defaults to window length
            nperseg = len(window)
    else:
        nperseg = int(nperseg)

    if noverlap is not None:
        noverlap = int(noverlap)

    # compute spectrogram of data
    freq, t, spg = signal.spectrogram(x, Fs, window, nperseg, noverlap, return_onesided=True)

    # get log10 power before binning
    ps = np.transpose(np.log10(spg))

    # Limit spectrogram to freq range of interest
    ps = ps[:, np.logical_and(freq >= flim[0], freq < flim[1])]
    freq = freq[np.logical_and(freq >= flim[0], freq < flim[1])]

    # Prepare bins for power. Min and max of bins determined by power cutoff percentage
    power_min, power_max = np.percentile(np.ndarray.flatten(ps), cutpct)
    power_bins = np.linspace(power_min, power_max, nbins + 1)

    # Compute histogram of power for each frequency
    spect_hist = np.zeros((len(ps[0]), nbins))
    for i in range(len(ps[0])):
        spect_hist[i], _ = np.histogram(ps[:, i], power_bins)
        spect_hist[i] = spect_hist[i] / sum(spect_hist[i])

    # flip them for sensible plotting direction
    spect_hist = np.transpose(spect_hist)
    spect_hist = np.flipud(spect_hist)

    return freq, power_bins, spect_hist


def plot_spectral_hist(freq, power_bins, spect_hist, psd_freq=None, psd=None):
    """
    Plot the spectral histogram.

    Parameters
    ----------
    freq : array_like, 1d
            Frequencies over which the histogram is calculated.
    power_bins : array_like, 1d
            Power bins within which histogram is aggregated.
    spect_hist : ndarray, 2d
            Spectral histogram to be plotted.
    psd_freq : array_like, 1d, optional
            Frequency axis of the PSD to be plotted.
    psd : array_like, 1d, optional
            PSD to be plotted over the histograms.
    """
    # automatically scale figure height based on number of bins
    plt.figure(figsize=(8, 12 * len(power_bins) / len(freq)))
    # plot histogram intensity as image and automatically adjust aspect ratio
    plt.imshow(spect_hist, extent=[freq[0], freq[-1], power_bins[0], power_bins[-1]], aspect='auto')
    plt.xlabel('Frequency (Hz)', fontsize=15)
    plt.ylabel('Log10 Power', fontsize=15)
    plt.colorbar(label='Probability')

    if psd is not None:
        # if a PSD is provided, plot over the histogram data
        plt.plot(psd_freq[np.logical_and(psd_freq >= freq[0], psd_freq <= freq[-1])], np.log10(
            psd[np.logical_and(psd_freq >= freq[0], psd_freq <= freq[-1])]), color='w', alpha=0.8)


def fit_slope(freq, psd, fit_frange, fit_excl=None, method='ols', plot_fit=False):
    """
    Fit PSD with straight line in log-log domain over the specified frequency range.

        Parameters
        ----------
        freq : array_like, 1d
                        Frequency axis of PSD
        psd : array_like, 1d
                        PSD to be fit over
        fit_frange : tuple, (start, end), Hz
                        Frequency range to be fit over, in Hz, inclusive on both ends.
        fit_excl : list of tuples, [(start, end), (start, end), ...], Hz, optional
                        Frequency ranges to be excluded from fit. Each element in list describes
                        the start and end of an exclusion zone.
        method : str, {'ols', 'RANSAC'}, optional
                        Line fitting method. Defaults to 'ols'
                        'ols' is ordinary least squares fit with polyfit.
                        'RANSAC' is iterative robust fit discarding outliers.
        plot_fit : bool, optional
                        If True, the PSD is plotted, along with the actual fitted PSD (excluding exclusion freqs),
                        as well as the fitted line itself. Defaults to False.

        Returns
        -------
        slope : float
                Slope of loglog fitted line, m in y = mx+b
        offset : float
                Offset of loglog fitted line, b in y = mx+b
    """

    # make a mask for included and excluded frequency regions
    fmask = np.zeros_like(freq)
    # get freq indices within the fit frequencies
    fmask[np.logical_and(freq >= fit_frange[0], freq <= fit_frange[1])] = 1
    # discard freq indices within the exclusion frequencies
    if fit_excl is not None:
        # if a tuple is given, convert it to a list
        if type(fit_excl) is tuple:
            fit_excl = [fit_excl]

        for exc_frange in fit_excl:
            fmask[np.logical_and(freq >= exc_frange[0], freq <= exc_frange[1])] = 0

    # grab the psd and freqs to be fit over
    logf = np.log10(freq[fmask == 1])
    logpsd = np.log10(psd[fmask == 1])

    # fit line
    if method is 'ols':
        # solve regular least square
        slope, offset = np.polyfit(logf, logpsd, deg=1)

    elif method is 'RANSAC':
        lm = linear_model.RANSACRegressor(random_state=42)
        lm.fit(logf.reshape(-1, 1), logpsd.reshape(-1, 1))
        offset = lm.predict(0.)[0][0]
        slope = lm.estimator_.coef_[0][0]

    else:
        raise ValueError('Unknown PSD fitting method: %s' % method)

    if plot_fit:
        plt.figure(figsize=(5, 5))
        plt.plot(np.log10(freq), np.log10(psd), label='Whole PSD')
        plt.plot(logf, logpsd, '-o', label='Fitted PSD', alpha=0.4)
        plt.plot(logf, logf * slope + offset, '-k', label='Fit Line', lw=3)
        plt.legend()
        plt.xlabel('Log10 Frequency (Hz)', fontsize=15)
        plt.ylabel('Log10 Power (V^2/Hz)', fontsize=15)

    return slope, offset
