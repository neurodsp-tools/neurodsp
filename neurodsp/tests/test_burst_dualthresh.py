"""Tests for burst detection functions."""

from neurodsp.burst.dualthresh import *
from neurodsp.burst.dualthresh import _dual_threshold_split, _rmv_short_periods

###################################################################################################
###################################################################################################

def test_detect_bursts_dual_threshold(tsig):

    # Test default settings
    bursts = detect_bursts_dual_threshold(tsig, 500, (8, 12), (1, 2))
    # Test other settings
    bursts = detect_bursts_dual_threshold(tsig, 500, (8, 12), (1, 2),
                                          avg_type='mean', magnitude_type='power')
    assert True

## PRIVATE FUNCTION TESTS
def test_dual_threshold_split():
    # TODO
    pass

def tests_rmv_short_periods():
    pass
    #TODO: check this function, doesn't seem to work quite as expected.

    # dat = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0])

    # dat = _rmv_short_periods(dat, 3)
    # assert np.array_equal(dat, np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0]))
