"""Tests for neurodsp.burst.utils."""

import pytest

from neurodsp.burst.utils import *

###################################################################################################
###################################################################################################

@pytest.mark.parametrize('bursting, n_bursts, duration_mean, percent_burst',
                         [(np.array([False, False, True, True, False]), 1, 2, 40),
                          (np.array([True, False, False, True, False, True]), 3, 1, 50)])
def test_compute_burst_stats(bursting, n_bursts, duration_mean, percent_burst):

    stats = compute_burst_stats(bursting, 1)

    assert stats['n_bursts'] == n_bursts
    assert stats['duration_mean'] == duration_mean
    assert stats['percent_burst'] == percent_burst
