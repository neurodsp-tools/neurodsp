"""Tools for estimating time frequency properties, using hilbert methods."""

import numpy as np
from scipy.signal import hilbert

from neurodsp.filt import filter_signal
from neurodsp.utils.decorators import multidim
from neurodsp.utils.outliers import remove_nans, restore_nans
from neurodsp.filt.utils import infer_passtype, remove_filter_edges

###################################################################################################
###################################################################################################

@multidim()
def robust_hilbert(sig, increase_n=False):
    """Compute the hilbert transform, ignoring any boundaries that are NaN.

    Parameters
    ----------
    sig : 1d array
        Time series.
    increase_n : bool, optional, default: False
        If True, zeropad the signal to length the next power of 2 for the hilbert transform.
        This is because :func:`scipy.signal.hilbert` can be very slow for some signal lengths.

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


@multidim()
def phase_by_time(sig, fs, f_range=None, hilbert_increase_n=False,
                  remove_edges=True, **filter_kwargs):
    """Compute the instantaneous phase of a time series.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of float or None, optional default: None
        Filter range, in Hz, as (low, high). If None, no filtering is applied.
    hilbert_increase_n : bool, optional, default: False
        If True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because :func:`scipy.signal.hilbert` can be very slow for some lengths of x.
    remove_edges : bool, optional, default: True
        If True, replace samples that are within half of the filters length to the edge with np.nan.
        This removes edge artifacts from the filtered signal. Only used if `f_range` is defined.
    **filter_kwargs
        Keyword parameters to pass to `filter_signal`.

    Returns
    -------
    pha : 1d array
        Instantaneous phase time series.
    """

    if f_range:
        sig, filter_kernel = filter_signal(sig, fs, infer_passtype(f_range), f_range=f_range,
                                           remove_edges=False, return_filter=True, **filter_kwargs)

    pha = np.angle(robust_hilbert(sig, increase_n=hilbert_increase_n))

    if f_range and remove_edges:
        pha = remove_filter_edges(pha, len(filter_kernel))

    return pha


@multidim()
def amp_by_time(sig, fs, f_range=None, hilbert_increase_n=False,
                remove_edges=True, **filter_kwargs):
    """Compute the instantaneous amplitude of a time series.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of float or None, optional default: None
        Filter range, in Hz, as (low, high). If None, no filtering is applied.
    hilbert_increase_n : bool, optional, default: False
        If True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because :func:`scipy.signal.hilbert` can be very slow for some lengths of sig.
    remove_edges : bool, optional, default: True
        If True, replace samples that are within half of the filters length to the edge with np.nan.
        This removes edge artifacts from the filtered signal. Only used if `f_range` is defined.
    **filter_kwargs
        Keyword parameters to pass to `filter_signal`.

    Returns
    -------
    amp : 1d array
        Instantaneous amplitude time series.
    """

    if f_range:
        sig, filter_kernel = filter_signal(sig, fs, infer_passtype(f_range), f_range=f_range,
                                           remove_edges=False, return_filter=True, **filter_kwargs)

    amp = np.abs(robust_hilbert(sig, increase_n=hilbert_increase_n))

    if f_range and remove_edges:
        amp = remove_filter_edges(amp, len(filter_kernel))

    return amp


@multidim()
def freq_by_time(sig, fs, f_range=None, hilbert_increase_n=False,
                 remove_edges=True, **filter_kwargs):
    """Compute the instantaneous frequency of a time series.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of float or None, optional default: None
        Filter range, in Hz, as (low, high). If None, no filtering is applied.
    hilbert_increase_n : bool, optional, default: False
        If True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because :func:`scipy.signal.hilbert` can be very slow for some lengths of sig.
    remove_edges : bool, optional, default: True
        If True, replace samples that are within half of the filters length to the edge with np.nan.
        This removes edge artifacts from the filtered signal. Only used if `f_range` is defined.
    **filter_kwargs
        Keyword parameters to pass to `filter_signal`.

    Returns
    -------
    i_f : 1d array
        Instantaneous frequency time series.

    Notes
    -----
    This function assumes monotonic phase, so phase slips will be processed as high frequencies.
    """

    pha = phase_by_time(sig, fs, f_range, hilbert_increase_n,
                        remove_edges, **filter_kwargs)

    phadiff = np.diff(pha)
    phadiff[phadiff < 0] = phadiff[phadiff < 0] + 2 * np.pi

    i_f = fs * phadiff / (2 * np.pi)
    i_f = np.insert(i_f, 0, np.nan)

    return i_f
