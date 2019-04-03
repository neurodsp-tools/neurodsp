"""Utility functions for filtering."""

import numpy as np
from scipy.signal import freqz

from neurodsp.filt.checks import check_filter_definition

###################################################################################################
###################################################################################################

def infer_passtype(f_range):
    """Given frequency definition of a filter, infer the passtype.

    Parameters
    ----------
    f_range : tuple of (float, float)
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.

    Returns
    -------
    pass_type : str
        Which kind of filter pass_type is consistent with the frequency definition provided.

    Notes
    -----
    Assumes that a definition with two frequencies is a 'bandpass' (not 'bandstop').
    """

    if f_range[0] is None:
        pass_type = 'lowpass'
    elif f_range[1] is None:
        pass_type = 'highpass'
    else:
        pass_type = 'bandpass'

    # Check the inferred passtype & frequency definition is valid
    _ = check_filter_definition(pass_type, f_range)

    return pass_type


def compute_frequency_response(b_vals, a_vals, fs):
    """Compute the frequency response of a filter.

    Parameters
    ----------
    b_vals : 1d array
        B value filter coefficients for a filter.
    a_vals : 1d array
        A value filter coefficients for a filter.
    fs : float
        Sampling rate, in Hz.

    Returns
    -------
    f_db : 1d array
        Frequency vector corresponding to attenuation decibels, in Hz.
    db : 1d array
        Degree of attenuation for each frequency specified in f_db, in dB.
    """

    w_vals, h_vals = freqz(b_vals, a_vals)
    f_db = w_vals * fs / (2. * np.pi)
    db = 20 * np.log10(abs(h_vals))

    return f_db, db


def compute_pass_band(fs, pass_type, f_range):
    """Compute the pass bandwidth of a filter.

    Parameters
    ----------
    fs : float
        Sampling rate, in Hz.
    pass_type : {'bandpass', 'bandstop', 'lowpass', 'highpass'}
        Which kind of filter to apply:

        * 'bandpass': apply a bandpass filter
        * 'bandstop': apply a bandstop (notch) filter
        * 'lowpass': apply a lowpass filter
        * 'highpass' : apply a highpass filter
    f_range : tuple of (float, float) or float
        Cutoff frequency(ies) used for filter, specified as f_lo & f_hi.
        For 'bandpass' & 'bandstop', must be a tuple.
        For 'lowpass' or 'highpass', can be a float that specifies pass frequency, or can be
        a tuple and is assumed to be (None, f_hi) for 'lowpass', and (f_lo, None) for 'highpass'.

    Returns
    -------
    pass_bw : float
        The pass bandwidth of the filter.
    """

    f_lo, f_hi = check_filter_definition(pass_type, f_range)
    if pass_type in ['bandpass', 'bandstop']:
        pass_bw = f_hi - f_lo
    elif pass_type == 'highpass':
        pass_bw = compute_nyquist(fs) - f_lo
    elif pass_type == 'lowpass':
        pass_bw = f_hi

    return pass_bw


def compute_transition_band(f_db, db, low=-20, high=-3):
    """Compute transition bandwidth of a filter.

    Parameters
    ----------
    f_db : 1d array
        Frequency vector corresponding to attenuation decibels, in Hz.
    db : 1d array
        Degree of attenuation for each frequency specified in f_db, in dB.
    low : float, optional, default: -20
        The lower limit that defines the transition band, in dB.
    high : float, optional, default: -3
        The upper limit that defines the transition band, in dB.

    Returns
    -------
    transition_band : float
        The transition bandwidth of the filter.
    """

    # This gets the indices of transitions to the values in searched for range
    inds = np.where(np.diff(np.logical_and(db > low, db < high)))[0]
    # This steps through the indices, in pairs, selecting from the vector to select from
    transition_band = np.max([(b - a) for a, b in zip(f_db[inds[0::2]], f_db[inds[1::2]])])

    return transition_band


def compute_nyquist(fs):
    """Compute the nyquist frequency.

    Parameters
    ----------
    fs : float
        Sampling rate, in Hz.

    Returns
    -------
    float
        The nyquist frequency of a signal with the given sampling rate, in Hz.
    """

    return fs / 2.


def remove_filter_edges(sig, filt_len):
    """Drop the edges, by making NaN, from a filtered signal, to avoid edge artifacts.

    Parameters
    ----------
    sig : 1d array
        Filtered signal to have edge artifacts removed from.
    filt_len : int
        Length of the filter that was applied.

    Returns
    -------
    sig : 1d array
        Filter signal with edge artifacts switched to NaNs.
    """

    n_rmv = int(np.ceil(filt_len / 2))
    sig[:n_rmv] = np.nan
    sig[-n_rmv:] = np.nan

    return sig
