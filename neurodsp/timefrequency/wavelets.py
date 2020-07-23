"""Time-frequency decompositions using wavelets."""

import numpy as np
from scipy.signal import morlet

from neurodsp.utils.data import create_freqs
from neurodsp.utils.checks import check_n_cycles
from neurodsp.utils.decorators import multidim

###################################################################################################
###################################################################################################

@multidim()
def compute_wavelet_transform(sig, fs, freqs, n_cycles=7, scaling=0.5, norm='amp'):
    """Compute the time-frequency representation of a signal using morlet wavelets.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    freqs : 1d array or list of float
        If array, frequency values to estimate with morlet wavelets.
        If list, define the frequency range, as [freq_start, freq_stop, freq_step].
        The `freq_step` is optional, and defaults to 1. Range is inclusive of `freq_stop` value.
    n_cycles : float or 1d array
        Length of the filter, as the number of cycles for each frequency.
        If 1d array, this defines n_cycles for each frequency.
    scaling : float
        Scaling factor.
    norm : {'sss', 'amp'}, optional
        Normalization method:

        * 'sss' - divide by the square root of the sum of squares
        * 'amp' - divide by the sum of amplitudes

    Returns
    -------
    mwt : 2d array
        Time frequency representation of the input signal.

    Notes
    -----

    * This computes the continuous wavelet transform at the specified frequencies and
        along all shifts.

    Examples
    --------
    Compute a Morlet wavelet time-frequency representation of a signal:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> mwt = compute_wavelet_transform(sig, fs=500, freqs=[1, 30])
    """

    if isinstance(freqs, (tuple, list)):
        freqs = create_freqs(*freqs)
    n_cycles = check_n_cycles(n_cycles, len(freqs))

    mwt = np.zeros([len(freqs), len(sig)], dtype=complex)
    for ind, (freq, n_cycle) in enumerate(zip(freqs, n_cycles)):
        mwt[ind, :] = convolve_wavelet(sig, fs, freq, n_cycle, scaling, norm=norm)

    return mwt


@multidim()
def convolve_wavelet(sig, fs, freq, n_cycles=7, scaling=0.5, wavelet_len=None, norm='sss'):
    """Convolve a signal with a complex wavelet.

    Parameters
    ----------
    sig : 1d array
        Time series to filter.
    fs : float
        Sampling rate, in Hz.
    freq : float
        Center frequency of bandpass filter.
    n_cycles : float, optional, default: 7
        Length of the filter, as the number of cycles of the oscillation with specified frequency.
    scaling : float, optional, default: 0.5
        Scaling factor for the morlet wavelet.
    wavelet_len : int, optional
        Length of the wavelet. If defined, this overrides the freq and n_cycles inputs.
    norm : {'sss', 'amp'}, optional
        Normalization method:

        * 'sss' - divide by the square root of the sum of squares
        * 'amp' - divide by the sum of amplitudes

    Returns
    -------
    array
        Complex time series.

    Notes
    -----

    * The real part of the returned array is the filtered signal.
    * Taking np.abs() of output gives the analytic amplitude.
    * Taking np.angle() of output gives the analytic phase.

    Examples
    --------
    Convolve a complex wavelet with a simulated signal:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> cts = convolve_wavelet(sig, fs=500, freq=10)
    """

    if wavelet_len is None:
        wavelet_len = int(n_cycles * fs / freq)

    if wavelet_len > sig.shape[-1]:
        raise ValueError('The length of the wavelet is greater than the signal. Can not proceed.')

    morlet_f = morlet(wavelet_len, w=n_cycles, s=scaling)

    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    elif norm == 'amp':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f))
    else:
        raise ValueError('Not a valid wavelet normalization method.')

    mwt_real = np.convolve(sig, np.real(morlet_f), mode='same')
    mwt_imag = np.convolve(sig, np.imag(morlet_f), mode='same')

    return mwt_real + 1j * mwt_imag
