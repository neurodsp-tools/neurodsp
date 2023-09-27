"""Tests for neurodsp.utils.decorators."""

from pytest import raises

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

    # Test function gets applied normally to 1D input
    arr1d = np.array([1, 2, 3, 4])
    assert func(arr1d) == 10

    # Test function gets applied across dimensions for 2D input
    arr2d = np.array([[1, 2], [1, 2]])
    assert np.array_equal(func(arr2d), np.array([3, 3]))

    # Check error for input of unsupported dimension
    arr3d = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    with raises(ValueError):
        func(arr3d)
