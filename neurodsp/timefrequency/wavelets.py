"""Time-frequency decompositions using wavelets."""

import numpy as np
from scipy.signal import morlet

from neurodsp.utils.decorators import multidim

###################################################################################################
###################################################################################################

@multidim
def morlet_transform(sig, fs, freqs, n_cycles=7, scaling=0.5):
    """Calculate the time-frequency representation of a signal using morlet wavelets.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    freqs : 1d array
        Frequency values to estimate with morlet wavelets.
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

    sig_len = len(sig)
    freqs_len = len(freqs)
    mwt = np.zeros([sig_len, freqs_len], dtype=complex)

    for f_ind, freq in enumerate(freqs):
        mwt[:, f_ind] = morlet_convolve(sig, fs, freq, n_cycles, scaling)

    return mwt


@multidim
def morlet_convolve(sig, fs, freq, n_cycles=7, scaling=0.5, filt_len=None, norm='sss'):
    """Convolve a signal with a complex wavelet.

    Parameters
    ----------
    sig : 1d array
        Time series to filter.
    fs : float
        Sampling rate, in Hz.
    freq : float
        Center frequency of bandpass filter.
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
        raise ValueError('Number of cycles for morlet wavelets must be a positive number.')

    if filt_len is None:
        filt_len = n_cycles * fs / freq

    morlet_f = morlet(filt_len, w=n_cycles, s=scaling)

    # ToDo: there is an inconsistency between these methods, and the docs.
    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    elif norm == 'abs':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f))
    else:
        raise ValueError('Not a valid wavelet normalization method.')

    mwt_real = np.convolve(sig, np.real(morlet_f), mode='same')
    mwt_imag = np.convolve(sig, np.imag(morlet_f), mode='same')

    return mwt_real + 1j * mwt_imag
