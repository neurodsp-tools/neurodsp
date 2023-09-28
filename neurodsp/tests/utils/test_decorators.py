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

    # 3d input
    #   note: func(arr3d) will return (dima, dimb, 1), so add the last dim to .sum() with a reshape
    arr3d = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    assert np.array_equal(func(arr3d), arr3d.sum(axis=-1).reshape((*arr3d.shape[:-1], 1)))

    # 4d input
    arr4d = np.random.rand(2, 3, 4, 5)
    assert np.array_equal(func(arr4d), arr4d.sum(axis=-1).reshape((*arr4d.shape[:-1], 1)))

    # 2d return shape (e.g. compute_spectrum)
    @multidim(select=[0])
    def func(sig):
        return np.arange(3), np.random.rand(3)

    freqs, powers = func(arr3d)
    assert np.array_equal(freqs, np.arange(3))
    assert powers.shape == (*arr3d.shape[:-1], 3)
