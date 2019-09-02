"""Test the sliding window matching function."""

from neurodsp.tests.settings import FS

from neurodsp.rhythm.swm import *

###################################################################################################
###################################################################################################

def test_sliding_window_matching(tsig):

    win_len, win_spacing = 1, 0.5

    pattern, starts, costs = sliding_window_matching(tsig, FS, win_len, win_spacing)
    assert pattern.shape[-1] == int(FS * win_len)

def test_sliding_window_matching_2d(tsig2d):

    win_len, win_spacing = 1, 0.5

    pattern, starts, costs = sliding_window_matching(tsig2d, FS, win_len, win_spacing)
    assert pattern.shape[-1] == int(FS * win_len)
