"""Tests for data related utility functions."""

from numpy.testing import assert_equal

from neurodsp.utils.data import *

###################################################################################################
###################################################################################################

def test_create_times():

    times = create_times(1, 10)
    assert_equal(times, np.arange(0, 1, 1/10))
