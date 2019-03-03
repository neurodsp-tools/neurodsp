"""Frequency domain analysis of neural signals: creating PSD, fitting 1/f, spectral histograms."""

import numpy as np
from scipy import signal

###################################################################################################
###################################################################################################

def compute_spectrum(sig, fs, method='mean', window='hann', nperseg=None,
                     noverlap=None, filt_len=1., f_lim=None, spg_outlier_pct=0.):
    """Estimate the power spectral density (PSD) of a time series from short-time Fourier
    Transform (mean, median), or the entire signal's FFT smoothed (medfilt).

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
        If window is array_like, this array will be used as the window and its length must be nperseg.
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
                nperseg = int(fs)
            else:
                # window is an array, defaults to window length
                nperseg = len(window)
        else:
            nperseg = int(nperseg)

        if noverlap is not None:
            noverlap = int(noverlap)

        # Calculate the short time fourier transform with signal.spectrogram
        freqs, t_axis, spg = signal.spectrogram(sig, fs, window, nperseg, noverlap)

        # Pad data to 2D
        if len(sig.shape) == 1:
            sig = sig[None, :]

        numchan = sig.shape[0]

        # Throw out outliers if indicated
        if spg_outlier_pct > 0.:

            n_discard = int(np.ceil(len(t_axis) / 100. * spg_outlier_pct))
            n_keep = int(len(t_axis)-n_discard)
            spg_temp = np.zeros((numchan, len(freqs), n_keep))
            outlier_inds = np.zeros((numchan, n_discard))

            # Discard time windows with high total log powers, round up so it doesn't get a zero
            for chan in range(numchan):
                outlier_inds[chan, :] = np.argsort(np.mean(np.log10(spg[chan, :, :]), axis=0))[-n_discard:]
                spg_temp[chan, :, :] = np.delete(spg[chan], outlier_inds[chan, :], axis=-1)
            spg = spg_temp

        if method == 'mean':
            spectrum = np.mean(spg, axis=-1)
        elif method == 'median':
            spectrum = np.median(spg, axis=-1)

    elif method == 'medfilt':

        # Take the positive half of the spectrum since it's symmetrical
        FT = np.fft.fft(sig)[:int(np.ceil(len(sig) / 2.))]
        freqs = np.fft.fftfreq(len(sig), 1. / fs)[:int(np.ceil(len(sig) / 2.))]  # get freq axis

        # Convert median filter length from Hz to samples
        filt_len_samp = int(int(filt_len / (freqs[1] - freqs[0])) / 2 * 2 + 1)
        spectrum = signal.medfilt(np.abs(FT)**2. / (fs * len(sig)), filt_len_samp)

    else:
        raise ValueError('Unknown power spectrum method: %s' % method)

    if f_lim is not None:
        f_lim_ind = np.where(freqs > f_lim)[0][0]
        return freqs[:f_lim_ind], spectrum[..., :f_lim_ind]
    else:
        return freqs, spectrum


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
        If window is array_like, this array will be used as the window and its length must be nperseg.
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

    if nperseg is None:
        if isinstance(window, str) or isinstance(window, tuple):
            # window is a string or tuple, defaults to 1 second of data
            nperseg = int(fs)
        else:
            # window is an array, defaults to window length
            nperseg = len(window)
    else:
        nperseg = int(nperseg)

    if noverlap is not None:
        noverlap = int(noverlap)

    freqs, _, spg = signal.spectrogram(sig, fs, window, nperseg, noverlap)

    if outlier_pct is not None:

        # Discard time windows with high powers. Round up so it doesn't get a zero.
        discard = int(np.ceil(spg.shape[1] / 100. * outlier_pct))
        outlieridx = np.argsort(np.mean(np.log10(spg), axis=0))[:-discard]
        spg = spg[:, outlieridx]

    spect_cv = np.std(spg, axis=-1) / np.mean(spg, axis=-1)

    return freqs, spect_cv


def compute_scv_rs(sig, fs, window='hann', nperseg=None, noverlap=0, method='bootstrap', rs_params=None):
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
        If window is array_like, this array will be used as the window and its length must be nperseg.
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
        Parameters for resampling algorithm.
        If method is 'bootstrap', rs_params = (nslices, ndraws), defaults to (10% of slices, 100 draws).
        This specifies the number of slices per draw, and number of random draws.
        If method is 'rolling', rs_params = (nslices, nsteps), defaults to (10, 5).
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

    if nperseg is None:
        if isinstance(window, (str, tuple)):
            # window is a string or tuple, defaults to 1 second of data
            nperseg = int(fs)
        else:
            # window is an array, defaults to window length
            nperseg = len(window)
    else:
        nperseg = int(nperseg)

    if noverlap is not None:
        noverlap = int(noverlap)

    freqs, ts, spg = signal.spectrogram(sig, fs, window, nperseg, noverlap)

    if method == 'bootstrap':

        # Params are number of slices of STFT to compute SCV over & number of draws
        if rs_params is None:
            # defaults to draw 1/10 of STFT slices, 100 draws
            rs_params = (int(spg.shape[1] / 10.), 100)

        nslices, ndraws = rs_params
        spect_cv_rs = np.zeros((len(freqs), ndraws))
        for draw in range(ndraws):
            # repeated subsampling of spectrogram randomly, with replacement
            # between draws
            idx = np.random.choice(spg.shape[1], size=nslices, replace=False)
            spect_cv_rs[:, draw] = np.std(
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


def spectral_hist(sig, fs, window='hann', nperseg=None, noverlap=None,
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
        If window is array_like, this array will be used as the window and its length must be nperseg.
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

    # Set the nperseg, if not provided:
    #   If the window is a string or tuple, defaults to 1 second of data
    #   If the window is an array, defaults to window length
    if nperseg is None:
        if isinstance(window, (str, tuple)):
            nperseg = int(fs)
        else:
            nperseg = len(window)
    else:
        nperseg = int(nperseg)

    if noverlap is not None:
        noverlap = int(noverlap)

    # Compute spectrogram of data
    freqs, _, spg = signal.spectrogram(sig, fs, window, nperseg, noverlap, return_onesided=True)

    # Get log10 power before binning
    ps = np.transpose(np.log10(spg))

    # Limit spectrogram to freq range of interest
    ps = ps[:, np.logical_and(freqs >= f_lim[0], freqs < f_lim[1])]
    freqs = freqs[np.logical_and(freqs >= f_lim[0], freqs < f_lim[1])]

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


def morlet_transform(sig, freqs, fs, n_cycles=7, scaling=0.5):
    """Calculate the time-frequency representation of a signal using morlet wavelets.

    Parameters
    ----------
    sig : 1d array
        Time series.
    freqs : 1d array
        Frequency axis.
    fs : float
        Sampling rate, in Hz.
    n_cycles : float
        Length of the filter, as the number of cycles of the oscillation
        whose frequency is the center of the bandpass filter.
    scaling : float
        Scaling factor.

    Returns
    -------
    mwt : 2d array
        Time-frequency representation of signal sig.
    """

    if n_cycles <= 0:
        raise ValueError('Number of cycles in a filter must be a positive number.')

    sig_len = len(sig)
    freqs_len = len(freqs)
    mwt = np.zeros([sig_len, freqs_len], dtype=complex)

    for f_ind, freq in enumerate(freqs):
        mwt[:,f_ind] = morlet_convolve(sig, freq, fs, n_cycles, scaling)

    return mwt


def morlet_convolve(sig, freq, fs, n_cycles=7, scaling=0.5, filt_len=None, norm='sss'):
    """Convolve a signal with a complex wavelet.

    Parameters
    ----------
    sig : 1d array
        Time series to filter.
    freq : float
        Center frequency of bandpass filter.
    fs : float
        Sampling rate, in Hz.
    n_cycles : float, optional, default=7
        Length of the filter, as the number of cycles of the oscillation with specified frequency.
    scaling : float, optional, default=0.5
        Scaling factor for the morlet wavelet.
    filt_len : integer, optional
        Length of the filter. If not None, this overrides the freq and n_cycles inputs.
    norm : {'sss', 'amp'}, optional
        Normalization method:

        * 'sss' - divide by the sqrt of the sum of squares of points
        * 'amp' - divide by the sum of amplitudes divided by 2

    Returns
    -------
    array
        Complex time series.

    Notes
    -----

    * The real part of the returned array is the filtered signal.
    * Taking np.abs() of output gives the analytic amplitude.
    * Taking np.angle() of output gives the analytic phase.
    """

    if n_cycles <= 0:
        raise ValueError('Number of cycles in a filter must be a positive number.')

    if filt_len is None:
        filt_len = n_cycles * fs / freq

    morlet_f = signal.morlet(filt_len, w=n_cycles, s=scaling)

    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    elif norm == 'abs':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f))
    else:
        raise ValueError('Not a valid wavelet normalization method.')

    mwt_real = np.convolve(sig, np.real(morlet_f), mode='same')
    mwt_imag = np.convolve(sig, np.imag(morlet_f), mode='same')

    return mwt_real + 1j * mwt_imag


def rotate_powerlaw(f_axis, spectrum, delta_f, f_rotation=None):
    """Change the power law exponent of a power spectrum about an axis frequency.

    Parameters
    ----------
    f_axis : 1d array
        Frequency axis of input spectrum, in Hz. Must be same length as spectrum.
    spectrum : 1d array
        Power spectrum to be rotated.
    delta_f : float
        Change in power law exponent to be applied. Positive is counterclockwise
        rotation (flatten), negative is clockwise rotation (steepen).
    f_rotation : float, optional
        Axis of rotation frequency, in Hz, such that power at that frequency is unchanged
        by the rotation. Only matters if not further normalizing signal variance.
        If None, the transform normalizes to power at 1Hz by defaults.

    Returns
    -------
    rot_spectrum : 1d array
        Rotated spectrum.
    """

    ## Make the 1/f rotation mask
    # If starting freq is 0Hz, default power at 0Hz to old value because log
    #   will return inf. Use case occurs in simulating/manipulating time series.
    f_mask = np.zeros_like(f_axis)

    if f_axis[0] == 0.:
        f_mask[0] = 1.
        f_mask[1:] = 10**(np.log10(np.abs(f_axis[1:])) * (delta_f))
    # Otherwise, apply rotation to all frequencies.
    else:
        f_mask = 10**(np.log10(np.abs(f_axis)) * (delta_f))

    if f_rotation is not None:

        # Normalize power at rotation frequency
        if f_rotation < np.abs(f_axis).min() or f_rotation > np.abs(f_axis).max():
            raise ValueError('Rotation frequency not within frequency range.')

        f_mask = f_mask / f_mask[np.where(f_axis >= f_rotation)[0][0]]

    # Apply mask to rotate spectrum
    rot_spectrum = f_mask * spectrum

    return rot_spectrum
