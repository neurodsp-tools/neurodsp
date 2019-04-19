"""Data related utility functions for NeuroDSP."""

import numpy as np

###################################################################################################
###################################################################################################

def create_times(n_seconds, fs, start_val=0.):
    """Create an array of time indices.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    start_val : float, optional, default=0
        Starting value for the time definition.

    Returns
    -------
    1d array
        Time indices.
    """

    return np.arange(start_val, n_seconds, 1/fs)
