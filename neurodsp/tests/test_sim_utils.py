"""Tests for sim utils."""

from numpy.testing import assert_equal

from neurodsp.sim.utils import *

###################################################################################################
###################################################################################################

def test_create_times():

    times = create_times(1, 10)
    assert_equal(times, np.arange(0, 1, 1/10))
