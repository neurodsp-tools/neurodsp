"""Tests for neurodsp.utils.yielders."""

from collections.abc import Iterable

import numpy as np

from neurodsp.utils.yielders import *

###################################################################################################
###################################################################################################

def test_step_over_time(tsig):

    start = 0
    size = 100
    step = 50

    yielder = step_over_time(tsig, start, size, step)
    assert isinstance(yielder, Iterable)

    it0 = next(yielder)
    np.array_equal(it0, tsig[0:size])

    it1 = next(yielder)
    np.array_equal(it0, tsig[0+step:step+size])

    # Test non-zero start
    nzero_start = 25
    yielder2 = step_over_time(tsig, nzero_start, size, step)
    assert np.array_equal(next(yielder2), tsig[nzero_start:nzero_start+size])

    for csig in yielder:
        pass

def test_step_over_signals(tsig2d):

    yielder = step_over_signals(tsig2d)
    assert isinstance(yielder, Iterable)

    it0 = next(yielder)
    assert np.array_equal(tsig2d[0, :], it0)

    it1 = next(yielder)
    assert np.array_equal(tsig2d[1, :], it1)
