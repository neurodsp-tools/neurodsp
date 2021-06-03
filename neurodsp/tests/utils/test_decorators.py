"""Tests for neurodsp.utils.decorators."""

import numpy as np

from neurodsp.utils.decorators import *

###################################################################################################
###################################################################################################

def test_normalize():

    @normalize
    def func():
        return np.array([1, 2, 3, 4, 5])

    # Check that output of function gets normalized
    out = func()
    assert np.isclose(out.mean(), 0.)
    assert np.isclose(out.var(), 1.)

def test_multidim():

    @multidim(select=[])
    def func(sig):
        return np.sum(sig)

    # Check that the output of the func gets applied across dimensions
    out = func(np.array([[1, 2], [1, 2]]))
    assert np.array_equal(out, np.array([3, 3]))
