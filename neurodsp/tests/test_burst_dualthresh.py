"""Tests for burst detection functions."""

from neurodsp.tests.settings import FS

from neurodsp.burst.dualthresh import *
from neurodsp.burst.dualthresh import _dual_threshold_split, _rmv_short_periods

###################################################################################################
###################################################################################################

def test_detect_bursts_dual_threshold(tsig):

    # Test default settings
    bursts = detect_bursts_dual_threshold(tsig, FS, (8, 12), (1, 2))

    # Test other settings
    bursts = detect_bursts_dual_threshold(tsig, FS, (8, 12), (1, 2),
                                          avg_type='mean', magnitude_type='power')

def test_detect_bursts_dual_threshold_2d(tsig2d):

    bursts = detect_bursts_dual_threshold(tsig2d, FS, (8, 12), (1, 2))
    assert bursts.shape == tsig2d.shape
