"""Tests for burst detection functions."""

from neurodsp.burst.utils import *

###################################################################################################
###################################################################################################

def test_compute_burst_stats():

    bursts = np.array([False, False, True, True, False])

    stats = compute_burst_stats(bursts, 1)

    assert stats['n_bursts'] == 1
    assert stats['duration_mean'] == 2
    assert stats['percent_burst'] == 40.0
