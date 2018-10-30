"""Frequency domain analysis of neural signals: creating PSD, fitting 1/f, spectral histograms."""

import numpy as np
from scipy import signal

###################################################################################################
###################################################################################################

def psd(x, Fs, method='mean', window='hann', nperseg=None,
        noverlap=None, filtlen=1., flim=None, spg_outlierpct=0.):
    """
    Estimating the power spectral density (PSD) of a time series from short-time Fourier
    Transform (mean, median), or the entire signal's FFT smoothed (medfilt).

    Parameters
    -----------
    x : array_like 1d or 2d
        Time series of measurement values.
    Fs : float, Hz
        Sampling frequency of the x time series.
    method : { 'mean', 'median', 'medfilt'}, optional
        Methods to calculate the PSD. Defaults to 'mean'.
            'mean' is the same as Welch's method (mean of STFT).
            'median' uses median of STFT instead of mean to minimize outlier effect.
            'medfilt' filters the entire signals raw FFT with a median filter to smooth.
    The next 3 parameters are only relevant for method = {'mean', 'median'}
    window : str or tuple or array_like, optional
        Desired window to use. Defaults to a Hann window.
            See scipy.signal.get_window for a list of windows and required parameters.
            If window is array_like, this array will be used as the window and its length must be nperseg.
    nperseg : int, optional
        Length of each segment, in number of samples. Defaults to None.
            If None, and window is str or tuple, is set to 1 second of data.
            If None, and window is array_like, is set to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If None, noverlap = nperseg // 2. Defaults to None.
    filten : float, Hz, optional
        (For medfilt method) Length of median filter in Hz.
    flim : float, Hz, optional
        Maximum frequency to keep. Defaults to None, which keeps up to Nyquist.
    spg_outlierpct : float, (between 0 to 100)
        Percentage of spectrogram windows with the highest powers to discard prior to averaging.
        Useful for quickly eliminating potential outliers to compute PSD.

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
            if isinstance(window, (str, tuple)):
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
        freq, t_axis, spg = signal.spectrogram(x, Fs, window, nperseg, noverlap)

        # pad data to 2D
        if len(x.shape) == 1:
            x = x[None, :]

        numchan = x.shape[0]
        # throw out outliers if indicated
        if spg_outlierpct > 0.:
            n_discard = int(np.ceil(len(t_axis) / 100. * spg_outlierpct))
            n_keep = int(len(t_axis)-n_discard)
            spg_ = np.zeros((numchan, len(freq), n_keep))
            outlier_inds = np.zeros((numchan, n_discard))
            for chan in range(numchan):
                # discard time windows with high total log powers, round up so it doesn't get a zero
                outlier_inds[chan, :] = np.argsort(np.mean(np.log10(spg[chan, :, :]), axis=0))[-n_discard:]
                spg_[chan, :, :] = np.delete(spg[chan], outlier_inds[chan, :], axis=-1)
            spg = spg_

        if method == 'mean':
            Pxx = np.mean(spg, axis=-1)
        elif method == 'median':
            Pxx = np.median(spg, axis=-1)

    elif method == 'medfilt':
        # median filtered FFT spectrum
        # take the positive half of the spectrum since it's symmetrical
        FT = np.fft.fft(x)[:int(np.ceil(len(x) / 2.))]
        freq = np.fft.fftfreq(
            len(x), 1. / Fs)[:int(np.ceil(len(x) / 2.))]  # get freq axis
        # convert median filter length from Hz to samples
        filtlen_samp = int(int(filtlen / (freq[1] - freq[0])) / 2 * 2 + 1)
        Pxx = signal.medfilt(np.abs(FT)**2. / (Fs * len(x)), filtlen_samp)

    else:
        raise ValueError('Unknown PSD method: %s' % method)

    if flim is not None:
        flim_ind = np.where(freq>flim)[0][0]
        return freq[:flim_ind], Pxx[...,:flim_ind]
    else:
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
        Desired window to use. Defaults to a Hann window.
            See scipy.signal.get_window for a list of windows and required parameters.
            If window is array_like, this array will be used as the window and its length must be nperseg.
    nperseg : int, optional
        Length of each segment, in number of samples. Defaults to None.
            If None, and window is str or tuple, is set to 1 second of data.
            If None, and window is array_like, is set to the length of the window.
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
        if isinstance(window, str) or isinstance(window, tuple):
            # window is a string or tuple, defaults to 1 second of data
            nperseg = int(Fs)
        else:
            # window is an array, defaults to window length
            nperseg = len(window)
    else:
        nperseg = int(nperseg)

    if noverlap is not None:
        noverlap = int(noverlap)

    freq, _, spg = signal.spectrogram(x, Fs, window, nperseg, noverlap)
    if outlierpct is not None:
        # discard time windows with high powers
        # round up so it doesn't get a zero
        discard = int(np.ceil(spg.shape[1] / 100. * outlierpct))
        outlieridx = np.argsort(np.mean(np.log10(spg), axis=0))[:-discard]
        spg = spg[:, outlieridx]

    spectcv = np.std(spg, axis=-1) / np.mean(spg, axis=-1)
    return freq, spectcv


def scv_rs(x, Fs, window='hann', nperseg=None, noverlap=0, method='bootstrap', rs_params=None):
    """
    Resampled version of scv: instead of a single estimate of mean and standard deviation,
    the spectrogram is resampled, either randomly (bootstrap) or time-stepped (rolling).

    Parameters
    -----------
    x : array_like 1d
        Time series of measurement values
    Fs : float, Hz
        Sampling frequency of the x time series.
    window : str or tuple or array_like, optional
        Desired window to use. Defaults to a Hann window.
            See scipy.signal.get_window for a list of windows and required parameters.
            If window is array_like, this array will be used as the window and its length must be nperseg.
    nperseg : int, optional
        Length of each segment, in number of samples. Defaults to None.
            If None, and window is str or tuple, is set to 1 second of data.
            If None, and window is array_like, is set to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. Defaults to 0 for independence.
    method : {'bootstrap', 'rolling'}, optional
        Method of resampling. Defaults to 'bootstrap'.
            'bootstrap' randomly samples a subset of the spectrogram repeatedly.
            'rolling' takes the rolling window scv.
    rs_params : tuple, (int, int), optional
        Parameters for resampling algorithm.
        If method is 'bootstrap', rs_params = (nslices, ndraws), defaults to (10% of slices, 100 draws).
            This specifies the number of slices per draw, and number of random draws,
        If method is 'rolling', rs_params = (nslices, nsteps), defaults to (10, 5).
            This specifies the number of slices per draw, and number of slices to step forward,

    Returns
    -------
    freq : ndarray
        Array of sample frequencies.
    t_inds : ndarray
        Array of time indices, for 'rolling' resampling. If 'bootstrap', t_inds = None.
    spectcv_rs : ndarray
        Resampled spectral coefficient of variation.
    """

    if nperseg is None:
        if isinstance(window, (str, tuple)):
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
    freq, ts, spg = signal.spectrogram(x, Fs, window, nperseg, noverlap)

    if method == 'bootstrap':
        # params are number of slices of STFT to compute SCV over, and number
        # of draws
        if rs_params is None:
            # defaults to draw 1/10 of STFT slices, 100 draws
            rs_params = (int(spg.shape[1] / 10.), 100)

        nslices, ndraws = rs_params
        spectcv_rs = np.zeros((len(freq), ndraws))
        for draw in range(ndraws):
            # repeated subsampling of spectrogram randomly, with replacement
            # between draws
            idx = np.random.choice(spg.shape[1], size=nslices, replace=False)
            spectcv_rs[:, draw] = np.std(
                spg[:, idx], axis=-1) / np.mean(spg[:, idx], axis=-1)

        t_inds = None  # no time component, return nothing

    elif method == 'rolling':
        # params are number of slices of STFT to compute SCV over, and number
        # of slices to roll forward
        if rs_params is None:
            # defaults to 10 STFT slices, move forward by 5 slices
            rs_params = (10, 5)

        nslices, nsteps = rs_params
        outlen = int(np.ceil((spg.shape[1] - nslices) / float(nsteps))) + 1
        spectcv_rs = np.zeros((len(freq), outlen))
        for ind in range(outlen):
            curblock = spg[:, nsteps * ind:nslices + nsteps * ind]
            spectcv_rs[:, ind] = np.std(
                curblock, axis=-1) / np.mean(curblock, axis=-1)

        t_inds = ts[0::nsteps]  # grab the time indices from the spectrogram

    else:
        raise ValueError('Unknown resampling method: %s' % method)

    return freq, t_inds, spectcv_rs


def spectral_hist(x, Fs, window='hann', nperseg=None, noverlap=None,
                  nbins=50, flim=(0., 100.), cutpct=(0., 100.)):
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
        Desired window to use. Defaults to a Hann window.
            See scipy.signal.get_window for a list of windows and required parameters.
            If window is array_like, this array will be used as the window and its length must be nperseg.
    nperseg : int, optional
        Length of each segment, in number of samples. Defaults to None.
            If None, and window is str or tuple, is set to 1 second of data.
            If None, and window is array_like, is set to the length of the window.
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
        if isinstance(window, (str, tuple)):
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
    freq, _, spg = signal.spectrogram(
        x, Fs, window, nperseg, noverlap, return_onesided=True)

    # get log10 power before binning
    ps = np.transpose(np.log10(spg))

    # Limit spectrogram to freq range of interest
    ps = ps[:, np.logical_and(freq >= flim[0], freq < flim[1])]
    freq = freq[np.logical_and(freq >= flim[0], freq < flim[1])]

    # Prepare bins for power. Min and max of bins determined by power cutoff
    # percentage
    power_min, power_max = np.percentile(np.ndarray.flatten(ps), cutpct)
    power_bins = np.linspace(power_min, power_max, nbins + 1)

    # Compute histogram of power for each frequency
    spect_hist = np.zeros((len(ps[0]), nbins))
    for ind in range(len(ps[0])):
        spect_hist[ind], _ = np.histogram(ps[:, ind], power_bins)
        spect_hist[ind] = spect_hist[ind] / sum(spect_hist[ind])

    # flip them for sensible plotting direction
    spect_hist = np.transpose(spect_hist)
    spect_hist = np.flipud(spect_hist)

    return freq, power_bins, spect_hist


def morlet_transform(x, f0s, Fs, w=7, s=.5):
    """Calculate the time-frequency representation using morlet wavelets.

    Parameters
    ----------
    x : array
        time series
    f0s : array
        frequency axis
    Fs : float
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles of the oscillation
        whose frequency is the center of the bandpass filter
    s : float
        Scaling factor

    Returns
    -------
    mwt : 2-D array
        time-frequency representation of signal x
    """

    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    T = len(x)
    F = len(f0s)
    mwt = np.zeros([F, T], dtype=complex)
    for f in range(F):
        mwt[f] = morlet_convolve(x, f0s[f], Fs, w=w, s=s)

    return mwt


def morlet_convolve(x, f0, Fs, w=7, s=.5, M=None, norm='sss'):
    """
    Convolve a signal with a complex wavelet
    The real part is the filtered signal
    Taking np.abs() of output gives the analytic amplitude
    Taking np.angle() of output gives the analytic phase

    Parameters
    ----------
    x : array
        Time series to filter
    f0 : float
        Center frequency of bandpass filter
    Fs : float
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles of the oscillation
        with frequency f0
    s : float
        Scaling factor for the morlet wavelet
    M : integer
        Length of the filter. Overrides the f0 and w inputs
    norm : string
        Normalization method
        'sss' - divide by the sqrt of the sum of squares of points
        'amp' - divide by the sum of amplitudes divided by 2

    Returns
    -------
    x_trans : array
        Complex time series
    """

    if w <= 0:
        raise ValueError('Number of cycles in a filter must be a positive number.')

    if M is None:
        M = w * Fs / f0

    morlet_f = signal.morlet(M, w=w, s=s)

    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    elif norm == 'abs':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f))
    else:
        raise ValueError('Not a valid wavelet normalization method.')

    mwt_real = np.convolve(x, np.real(morlet_f), mode='same')
    mwt_imag = np.convolve(x, np.imag(morlet_f), mode='same')

    return mwt_real + 1j * mwt_imag


def rotate_powerlaw(f_axis, psd, delta_f, f_rotation=None):
    """Change the power law exponent of a PSD about an axis frequency.

    Parameters
    ----------
    f_axis : 1d array, Hz
        Frequency axis of input PSD. Must be same length as psd.
    psd : 1d array
        Power spectrum to be rotated.
    delta_f : float
        Change in power law exponent to be applied. Positive is counterclockwise
        rotation (flatten), negative is clockwise rotation (steepen).
    f_rotation : float, Hz, optional
        Axis of rotation frequency, such that power at that frequency is unchanged
        by the rotation. Only matters if not further normalizing signal variance.
        If None, the transform normalizes to power at 1Hz by defaults.

    Returns
    -------
    1d array
        Rotated psd.
    """

    # make the 1/f rotation mask
    f_mask = np.zeros_like(f_axis)
    if f_axis[0] == 0.:
        # If starting freq is 0Hz, default power at 0Hz to old value because log
        # will return inf. Use case occurs in simulating/manipulating time series.
        f_mask[0] = 1.
        f_mask[1:] = 10**(np.log10(np.abs(f_axis[1:])) * (delta_f))
    else:
        # Otherwise, apply rotation to all frequencies.
        f_mask = 10**(np.log10(np.abs(f_axis)) * (delta_f))

    if f_rotation is not None:
        # normalize power at rotation frequency
        if f_rotation < np.abs(f_axis).min() or f_rotation > np.abs(f_axis).max():
            raise ValueError('Rotation frequency not within frequency range.')

        f_mask = f_mask / f_mask[np.where(f_axis >= f_rotation)[0][0]]

    # apply mask
    return f_mask * psd
