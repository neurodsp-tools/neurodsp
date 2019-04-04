"""Test functions for timefrequency analysis."""

from neurodsp.timefrequency.tf import *

###################################################################################################
###################################################################################################

def test_phase_by_time(tsig):

    out = phase_by_time(tsig, 500, (8, 12))
    assert True

def test_amp_by_time(tsig):

    out = amp_by_time(tsig, 500, (8, 12))
    assert True

def test_freq_by_time(tsig):

    out = freq_by_time(tsig, 500, (8, 12))
    assert True
