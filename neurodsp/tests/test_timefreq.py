"""Test functions for the time-frequency analysis module."""

import numpy as np

from neurodsp.timefrequency import *

###################################################################################################
###################################################################################################

def test_phase_by_time():
    pass

def test_amp_by_time():
    pass

def test_freq_by_time():
    pass

## PRIVATE FUNCTIONS

def test_robust_hilbert():

    # Generate a signal with NaNs
    fs, n_points, n_nans = 100, 1000, 10
    sig = np.random.randn(n_points)
    sig[0:n_nans] = np.nan

    # Check has correct number of nans (not all nan), without increase_n
    hilb_sig = robust_hilbert(sig)
    assert sum(np.isnan(hilb_sig)) == n_nans

    # Check has correct number of nans (not all nan), with increase_n
    hilb_sig = robust_hilbert(sig, True)
    assert sum(np.isnan(hilb_sig)) == n_nans
