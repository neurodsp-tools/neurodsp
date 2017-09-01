"""
timefrequency.py
Tools for estimating properties of a neural oscillation over time
"""

import numpy as np
import scipy as sp
from scipy import signal
import math
import neurodsp


def phase_by_time(x, Fs, f_range,
                  filter_fn=None, filter_kwargs=None,
                  hilbert_increase_N=False):
    """
    Calculate the phase time series of a neural oscillation

    Parameters
    ----------
    x : array-like, 1d
        Time series
    f_range : (low, high), Hz
        Frequency range
    Fs : float, Hz
        Sampling rate
    filter_fn : function
        The filtering function, with api:
        `filterfn(x, Fs, pass_type, f_lo, f_hi, remove_edge_artifacts=True)
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    hilbert_increase_N : bool
        if True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of x

    Returns
    -------
    pha : array-like, 1d
        Time series of phase
    """
    # Set default filtering parameters
    if filter_fn is None:
        filter_fn = neurodsp.filter
    if filter_kwargs is None:
        filter_kwargs = {}
    # Filter signal
    x_filt = filter_fn(x, Fs, 'bandpass', f_lo=f_range[0], f_hi=f_range[1], remove_edge_artifacts=False, **filter_kwargs)
    # Compute phase time series
    pha = np.angle(_hilbert_ignore_nan(x_filt, hilbert_increase_N=hilbert_increase_N))
    return pha


def amp_by_time(x, Fs, f_range,
                filter_fn=None, filter_kwargs=None,
                hilbert_increase_N=False):
    """
    Calculate the amplitude time series

    Parameters
    ----------
    x : array-like, 1d
        Time series
    f_range : (low, high), Hz
        The frequency filtering range
    Fs : float, Hz
        Sampling rate
    filter_fn : function
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
        Must have the same API as filt.bandpass
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    hilbert_increase_N : bool
        if True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of x

    Returns
    -------
    amp : array-like, 1d
        Time series of amplitude
    """
    # Set default filtering parameters
    if filter_fn is None:
        filter_fn = neurodsp.filter
    if filter_kwargs is None:
        filter_kwargs = {}
    # Filter signal
    x_filt = filter_fn(x, Fs, 'bandpass', f_lo=f_range[0], f_hi=f_range[1], remove_edge_artifacts=False, **filter_kwargs)
    # Compute amplitude time series
    amp = np.abs(_hilbert_ignore_nan(x_filt, hilbert_increase_N=hilbert_increase_N))
    return amp


def freq_by_time(x, Fs, f_range):
    '''
    Estimate the instantaneous frequency at each sample

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        sampling rate
    f_range : (low, high), Hz
        frequency range for filtering

    Returns
    -------
    i_f : float
        estimate instantaneous frequency for each sample in 'x'

    Notes
    -----
    * This function assumes monotonic phase, so
    a phase slip will be processed as a very high frequency
    '''
    pha = phase_by_time(x, Fs, f_range)
    phadiff = np.diff(pha)
    phadiff[phadiff < 0] = phadiff[phadiff < 0] + 2 * np.pi
    i_f = Fs * phadiff / (2 * np.pi)
    i_f = np.insert(i_f, 0, np.nan)
    return i_f


def _hilbert_ignore_nan(x, hilbert_increase_N=False):
    """
    Compute the hilbert transform of x.
    Ignoring the boundaries of x that are filled with NaN
    """
    # Extract the signal that is not nan
    first_nonan = np.where(~np.isnan(x))[0][0]
    last_nonan = np.where(~np.isnan(x))[0][-1] + 1
    x_nonan = x[first_nonan:last_nonan]

    # Compute hilbert transform of signal without nans
    if hilbert_increase_N:
        N = len(x_nonan)
        N2 = 2**(int(math.log(N, 2)) + 1)
        x_hilb_nonan = signal.hilbert(x_nonan, N2)
    else:
        x_hilb_nonan = signal.hilbert(x_nonan)

    # Fill in output hilbert with nans on edges
    x_hilb = np.ones(len(x), dtype=complex) * np.nan
    x_hilb[first_nonan:last_nonan] = x_hilb_nonan
    return x_hilb
