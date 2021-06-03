"""Tests for neurodsp.rhythm.swm (sliding window matching)."""

from neurodsp.tests.settings import FS

from neurodsp.rhythm.swm import *

###################################################################################################
###################################################################################################

def test_sliding_window_matching(tsig):

    win_len, win_spacing = 1, 0.5

    windows, starts = sliding_window_matching(tsig, FS, win_len, win_spacing, var_thresh=0.1)
    assert windows.shape[-1] == int(FS * win_len)

def test_sliding_window_matching_2d(tsig2d):

    win_len, win_spacing = 1, 0.5

    windows, starts = sliding_window_matching(tsig2d, FS, win_len, win_spacing, var_thresh=0.1)
    assert windows.shape[-1] == int(FS * win_len)
