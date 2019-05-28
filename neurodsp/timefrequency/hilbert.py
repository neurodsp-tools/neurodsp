"""Tools for estimating time frequency properties, using hilbert methods."""

import numpy as np
from scipy.signal import hilbert

from neurodsp.filt import filter_signal
from neurodsp.utils.decorators import multidim
from neurodsp.utils.outliers import remove_nans, restore_nans
from neurodsp.filt.utils import infer_passtype, remove_filter_edges

###################################################################################################
###################################################################################################

@multidim
def robust_hilbert(sig, increase_n=False):
    """Compute the hilbert transform, ignoring the boundaries of that are filled with NaN.

    Parameters
    ----------
    sig : 1d array
        Time series.
    increase_n : bool, optional, default: False
        If True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of x.

    Returns
    -------
    sig_hilb : 1d array
        The hilbert transform of the input signal.
    """

    # Extract the signal that is not nan
    sig_nonan, sig_nans = remove_nans(sig)

    # Compute hilbert transform of signal without nans
    if increase_n:
        sig_len = len(sig_nonan)
        n_components = 2**(int(np.log2(sig_len)) + 1)
        sig_hilb_nonan = hilbert(sig_nonan, n_components)[:sig_len]
    else:
        sig_hilb_nonan = hilbert(sig_nonan)

    # Fill in output hilbert with nans on edges
    sig_hilb = restore_nans(sig_hilb_nonan, sig_nans, dtype=complex)

    return sig_hilb


@multidim
def phase_by_time(sig, fs, f_range, hilbert_increase_n=False, remove_edges=True, **filter_kwargs):
    """Calculate the phase time series of a neural oscillation.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of float
        Frequency range, in Hz, as (low, high).
    hilbert_increase_n : bool, optional, default: False
        If True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of x.
    remove_edges : bool, optional, default: True
        If True, replace the samples that are within half a kernel's length to
        the signal edge with np.nan.
    **filter_kwargs
        Keyword parameters to pass to `filter_signal`.

    Returns
    -------
    pha : 1d array
        Time series of phase.
    """

    sig_filt, kernel = filter_signal(sig, fs, infer_passtype(f_range), f_range=f_range,
                                     remove_edges=False, return_filter=True, **filter_kwargs)

    pha = np.angle(robust_hilbert(sig_filt, increase_n=hilbert_increase_n))

    if remove_edges:
        pha = remove_filter_edges(pha, len(kernel))

    return pha


@multidim
def amp_by_time(sig, fs, f_range, hilbert_increase_n=False, remove_edges=True, **filter_kwargs):
    """Calculate the amplitude time series.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of float
        The frequency filtering range, in Hz, as (low, high).
    hilbert_increase_n : bool, optional, default: False
        If True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of sig.
    remove_edges : bool, optional, default: True
        If True, replace the samples that are within half a kernel's length to
        the signal edge with np.nan.
    **filter_kwargs
        Keyword parameters to pass to `filter_signal`.

    Returns
    -------
    amp : 1d array
        Time series of amplitude.
    """

    sig_filt, kernel = filter_signal(sig, fs, infer_passtype(f_range), f_range=f_range,
                                     remove_edges=False, return_filter=True, **filter_kwargs)

    amp = np.abs(robust_hilbert(sig_filt, increase_n=hilbert_increase_n))

    if remove_edges:
        amp = remove_filter_edges(amp, len(kernel))

    return amp


@multidim
def freq_by_time(sig, fs, f_range, hilbert_increase_n=False, remove_edges=True, **filter_kwargs):
    """Estimate the instantaneous frequency at each sample.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of float
        Frequency range for filtering, in Hz, as (low, high).
    hilbert_increase_n : bool, optional, default: False
        If True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of sig.
    remove_edges : bool, optional, default: True
        If True, replace the samples that are within half a kernel's length to
        the signal edge with np.nan.
    **filter_kwargs
        Keyword parameters to pass to `filter_signal`.

    Returns
    -------
    i_f : float
        Estimated instantaneous frequency for each sample in 'sig'.

    Notes
    -----
    This function assumes monotonic phase, so a phase slip will
    be processed as a very high frequency.
    """

    pha = phase_by_time(sig, fs, f_range, hilbert_increase_n,
                        remove_edges, **filter_kwargs)

    phadiff = np.diff(pha)
    phadiff[phadiff < 0] = phadiff[phadiff < 0] + 2 * np.pi

    i_f = fs * phadiff / (2 * np.pi)
    i_f = np.insert(i_f, 0, np.nan)

    return i_f
