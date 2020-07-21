"""Tools for estimating time frequency properties, using Hilbert methods."""

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
    """Compute the Hilbert transform, ignoring any boundaries that are NaN.

    Parameters
    ----------
    sig : 1d array
        Time series.
    increase_n : bool, optional, default: False
        If True, zero pad the signal's length to the next power of 2 for the Hilbert transform.
        This is because ``scipy.signal.hilbert`` can be very slow for some signal lengths.

    Returns
    -------
    sig_hilb : 1d array
        The analytic signal, of which the imaginary part is the Hilbert transform of the input.

    Examples
    --------
    Compute the analytic signal, using zero padding:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation': {'freq': 10}})
    >>> sig_hilb = robust_hilbert(sig, increase_n=True)
    """

    # Extract the signal that is not nan
    sig_nonan, sig_nans = remove_nans(sig)

    # Compute Hilbert transform of signal without nans
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
        If True, zero pad the signal's length to the next power of 2 for the Hilbert transform.
        This is because ``scipy.signal.hilbert`` can be very slow for some lengths of x.
    remove_edges : bool, optional, default: True
        If True, replace samples that are within half of the filter's length to the edge with nan.
        This removes edge artifacts from the filtered signal. Only used if `f_range` is defined.
    **filter_kwargs
        Keyword parameters to pass to `filter_signal`.

    Returns
    -------
    pha : 1d array
        Instantaneous phase time series.

    Examples
    --------
    Compute the instantaneous phase, for the alpha range:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation': {'freq': 10}})
    >>> pha = phase_by_time(sig, fs=500, f_range=(8, 12))
    """

    if f_range:
        sig, filter_kernel = filter_signal(sig, fs, infer_passtype(f_range),
                                           f_range=f_range, remove_edges=False,
                                           return_filter=True, **filter_kwargs)

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
        If True, zero pad the signal's length to the next power of 2 for the Hilbert transform.
        This is because ``scipy.signal.hilbert`` can be very slow for some lengths of sig.
    remove_edges : bool, optional, default: True
        If True, replace samples that are within half of the filter's length to the edge with nan.
        This removes edge artifacts from the filtered signal. Only used if `f_range` is defined.
    **filter_kwargs
        Keyword parameters to pass to `filter_signal`.

    Returns
    -------
    amp : 1d array
        Instantaneous amplitude time series.

    Examples
    --------
    Compute the instantaneous amplitude, for the alpha range:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> amp = amp_by_time(sig, fs=500, f_range=(8, 12))
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
        If True, zero pad the signal's length to the next power of 2 for the Hilbert transform.
        This is because ``scipy.signal.hilbert`` can be very slow for some lengths of sig.
    remove_edges : bool, optional, default: True
        If True, replace samples that are within half of the filter's length to the edge with nan.
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

    Examples
    --------
    Compute the instantaneous frequency, for the alpha range:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> instant_freq = freq_by_time(sig, fs=500, f_range=(8, 12))
    """

    # Calculate instantaneous phase, from which we can compute instantaneous frequency
    pha = phase_by_time(sig, fs, f_range, hilbert_increase_n, remove_edges, **filter_kwargs)

    # If filter edges were removed, temporarily drop nans for subsequent operations
    pha, nans = remove_nans(pha)

    # Differentiate the unwrapped phase, to estimate frequency as the rate of change of phase
    pha_unwrapped = np.unwrap(pha)
    pha_diff = np.diff(pha_unwrapped)

    # Convert differentiated phase to frequency
    i_f = fs * pha_diff / (2 * np.pi)

    # Prepend nan value to re-align with original signal (necessary due to differentiation)
    i_f = np.insert(i_f, 0, np.nan)

    # Restore nans, to re-align signal if filter edges were removed
    i_f = restore_nans(i_f, nans)

    return i_f
