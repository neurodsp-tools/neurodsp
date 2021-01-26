"""Data related utility functions."""

import numpy as np

###################################################################################################
###################################################################################################

def create_freqs(freq_start, freq_stop, freq_step=1):
    """Create an array of frequencies.

    Parameters
    ----------
    freq_start : float
        Starting value for the frequency definition.
    freq_stop : float
        Stopping value for the frequency definition, inclusive.
    freq_step : float, optional, default: 1
        Step value, for linearly spaced values between start and stop.

    Returns
    -------
    1d array
        Frequency indices.
    """

    return np.arange(freq_start, freq_stop + freq_step, freq_step)


def create_times(n_seconds, fs, start_val=0.):
    """Create an array of time indices.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    start_val : float, optional, default: 0
        Starting value for the time definition.

    Returns
    -------
    1d array
        Time indices.
    """

    return np.linspace(start_val, n_seconds, int(fs * n_seconds)+1)[:-1]


def create_samples(n_samples, start_val=0):
    """Create an array of sample indices.

    Parameters
    ----------
    n_seconds : int
        Signal duration, in seconds.
    start_val : int, optional, default: 0
        Starting value for the samples definition.

    Returns
    -------
    1d array
        Sample indices.
    """

    return np.arange(start_val, n_samples, 1)


def compute_nsamples(n_seconds, fs):
    """Calculate the number of samples for a given time definition.

    Parameters
    ----------
    n_seconds : int
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.

    Returns
    -------
    int
        The number of samples.

    Notes
    -----
    The result has to be rounded, in order to ensure that the number of samples is a whole number.

    The `int` function rounds down, by default, which is the convention across the module.
    """

    return int(n_seconds * fs)


def split_signal(sig, n_samples):
    """Split a signal into non-overlapping segments.

    Parameters
    ----------
    sig : 1d array
        Time series.
    n_samples : int
        The chunk size to split the signal into, in samples.

    Returns
    -------
    segs : 2d array
        The signal, split into segments, with shape [n_segment, segment_size].

    Notes
    -----
    If the signal does not divide evenly into the number of segments, this approach
    will truncate the signal, returning the maximum number of segments, and dropping
    any leftover samples.
    """

    n_segments = int(np.floor(len(sig) / float(n_samples)))
    segments = np.reshape(sig[:int(n_segments * n_samples)], (n_segments, int(n_samples)))

    return segments
