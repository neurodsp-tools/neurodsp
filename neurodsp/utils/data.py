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
    freqs : 1d array
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
    times : 1d array
        Time indices.
    """

    return np.linspace(start_val, n_seconds+start_val, compute_nsamples(n_seconds, fs)+1)[:-1]


def create_samples(n_samples, start_val=0):
    """Create an array of sample indices.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    start_val : int, optional, default: 0
        Starting value for the samples definition.

    Returns
    -------
    samples : 1d array
        Sample indices.
    """

    return np.arange(start_val, n_samples, 1)


def compute_nsamples(n_seconds, fs):
    """Calculate the number of samples for a given time definition.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.

    Returns
    -------
    n_samples : int
        The number of samples.

    Notes
    -----
    The result has to be rounded, in order to ensure that the number of samples is a whole number.
    By convention, this rounds up, which is needed to ensure that cycles don't end up being shorter
    than expected, which can lead to shorter than expected signals, after concatenation.
    """

    return int(np.ceil(n_seconds * fs))


def compute_nseconds(sig, fs):
    """Compute the length, in time, of a signal.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Signal sampling rate, in Hz.

    Returns
    -------
    fs : float
        Signal duration, in seconds.
    """

    return len(sig) / fs


def compute_cycle_nseconds(freq, fs=None):
    """Compute the length, in seconds, for a single cycle at a particular frequency.

    Parameters
    ----------
    freq : float
        Oscillation frequency, in Hz.
    fs :  float, optional
        Sampling rate, in Hz.
        If provided, this is used to get a cycle length optimized for the sampling rate.

    Returns
    -------
    n_seconds : float
        The number of seconds of a single cycle at the specified frequency.

    Notes
    -----
    The rounding is used to get a value that works with the sampling rate.
    """

    if fs:
        n_seconds = int(np.ceil(fs / freq)) / fs
    else:
        n_seconds = 1 / freq

    return n_seconds


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
    segments : 2d array
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
