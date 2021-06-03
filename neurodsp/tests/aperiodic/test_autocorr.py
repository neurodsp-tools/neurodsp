"""Tests for neurodsp.aperiodic.autocorr."""

from neurodsp.aperiodic.autocorr import *

###################################################################################################
###################################################################################################

def test_compute_autocorr(tsig):

    max_lag = 500
    timepoints, autocorrs = compute_autocorr(tsig, max_lag=max_lag)
    assert len(timepoints) == len(autocorrs) == max_lag + 1
