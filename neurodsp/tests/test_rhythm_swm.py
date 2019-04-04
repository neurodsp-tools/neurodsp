"""Test the sliding window matching function."""

from neurodsp.rhythm.swm import *

###################################################################################################
###################################################################################################

def test_sliding_window_matching(tsig):

    sliding_window_matching(tsig, 100, 1, 0.5)
    assert True

## PRIVATE FUNCTIONS

def test_compute_cost():
    pass

def test_find_new_windowidx():
    pass
