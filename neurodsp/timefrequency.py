"""Tools for estimating properties of a neural oscillation over time."""

import math

import numpy as np
from scipy import signal

import neurodsp

###################################################################################################
###################################################################################################

def phase_by_time(x, Fs, f_range, filter_fn=None, filter_kwargs=None, hilbert_increase_N=False,
                  remove_edge_artifacts=True):
    """Calculate the phase time series of a neural oscillation.

    Parameters
    ----------
    x : array-like, 1d
        Time series
    Fs : float, Hz
        Sampling rate
    f_range : (low, high), Hz
        Frequency range
    filter_fn : function, optional
        The filtering function, with api:
        `filterfn(x, Fs, pass_type, fc, remove_edge_artifacts=True)
    filter_kwargs : dict, optional
        Keyword parameters to pass to `filterfn(.)`
    hilbert_increase_N : bool, optional
        if True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of x
    remove_edge_artifacts : bool, optional
        if True, replace the samples that are within half a kernel's length to
        the signal edge with np.nan

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
    x_filt, kernel = filter_fn(x, Fs, 'bandpass', fc=f_range,
                               remove_edge_artifacts=False, 
                               return_kernel=True, **filter_kwargs)

    # Compute phase time series
    pha = np.angle(_hilbert_ignore_nan(x_filt, hilbert_increase_N=hilbert_increase_N))
    # Remove edge artifacts
    if remove_edge_artifacts:
        N_rmv = int(np.ceil(len(kernel) / 2))
        pha[:N_rmv] = np.nan
        pha[-N_rmv:] = np.nan
    return pha


def amp_by_time(x, Fs, f_range, filter_fn=None, filter_kwargs=None, hilbert_increase_N=False,
                remove_edge_artifacts=True):
    """Calculate the amplitude time series.

    Parameters
    ----------
    x : array-like, 1d
        Time series
    Fs : float, Hz
        Sampling rate
    f_range : (low, high), Hz
        The frequency filtering range
    filter_fn : function, optional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
        Must have the same API as filt.bandpass
    filter_kwargs : dict, optional
        Keyword parameters to pass to `filterfn(.)`
    hilbert_increase_N : bool, optional
        if True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of x
    remove_edge_artifacts : bool, optional
        if True, replace the samples that are within half a kernel's length to
        the signal edge with np.nan

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
    x_filt, kernel = filter_fn(x, Fs, 'bandpass', fc=f_range,
                               remove_edge_artifacts=False, 
                               return_kernel=True, **filter_kwargs)

    # Compute amplitude time series
    amp = np.abs(_hilbert_ignore_nan(x_filt, hilbert_increase_N=hilbert_increase_N))
    # Remove edge artifacts
    if remove_edge_artifacts:
        N_rmv = int(np.ceil(len(kernel) / 2))
        amp[:N_rmv] = np.nan
        amp[-N_rmv:] = np.nan
    return amp


def freq_by_time(x, Fs, f_range, filter_fn=None, filter_kwargs=None, hilbert_increase_N=False,
                 remove_edge_artifacts=True):
    """Estimate the instantaneous frequency at each sample.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        sampling rate
    f_range : (low, high), Hz
        frequency range for filtering
    filter_fn : function, optional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
        Must have the same API as filt.bandpass
    filter_kwargs : dict, optional
        Keyword parameters to pass to `filterfn(.)`
    hilbert_increase_N : bool, optional
        if True, zeropad the signal to length the next power of 2 when doing the hilbert transform.
        This is because scipy.signal.hilbert can be very slow for some lengths of x
    remove_edge_artifacts : bool, optional
        if True, replace the samples that are within half a kernel's length to
        the signal edge with np.nan

    Returns
    -------
    i_f : float
        estimate instantaneous frequency for each sample in 'x'

    Notes
    -----
    * This function assumes monotonic phase, so
    a phase slip will be processed as a very high frequency
    """

    pha = phase_by_time(x, Fs, f_range, filter_fn=filter_fn,
                        filter_kwargs=filter_kwargs,
                        hilbert_increase_N=hilbert_increase_N,
                        remove_edge_artifacts=remove_edge_artifacts)
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
        x_hilb_nonan = signal.hilbert(x_nonan, N2)[:N]
    else:
        x_hilb_nonan = signal.hilbert(x_nonan)

    # Fill in output hilbert with nans on edges
    x_hilb = np.ones(len(x), dtype=complex) * np.nan
    x_hilb[first_nonan:last_nonan] = x_hilb_nonan

    return x_hilb
