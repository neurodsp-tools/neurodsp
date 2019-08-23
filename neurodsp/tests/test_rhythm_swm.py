"""Test the sliding window matching function."""

from neurodsp.tests.settings import FS

from neurodsp.rhythm.swm import *

###################################################################################################
###################################################################################################

def test_sliding_window_matching(tsig):

    pattern = sliding_window_matching(tsig, FS, 1, 0.5)
