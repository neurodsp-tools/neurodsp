"""Tests for neurodsp.utils.decorators."""

from pytest import raises

import numpy as np

from neurodsp.timefrequency import robust_hilbert
from neurodsp.filt import filter_signal
from neurodsp.aperiodic import compute_autocorr
from neurodsp.spectral import compute_spectrum

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

    # Test 1D inputs
    arr = np.array([[1, 2], [1, 2]])
    assert np.array_equal(func(arr), arr.sum(axis=-1))

    arr1d = np.array([1, 2, 3, 4])
    assert func(arr1d) == arr1d.sum()

    # Test 2D inputs
    arr2d = np.array([[1, 2], [1, 2]])
    assert np.array_equal(func(arr2d), arr2d.sum(axis=-1))

    # Test 3D inputs
    arr3d = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    assert np.array_equal(func(arr3d), arr3d.sum(axis=-1))

    # Test 4D inputs
    arr4d = np.random.rand(2, 3, 4, 5)
    assert np.array_equal(func(arr4d), arr4d.sum(axis=-1))

    # Test 2d return shape (e.g. compute_spectrum)
    @multidim(select=[0])
    def func2(sig):
        return np.arange(3), np.random.rand(3)

    freqs, powers = func2(arr3d)
    assert np.array_equal(freqs, np.arange(3))
    assert powers.shape == (*arr3d.shape[:-1], 3)

    # Accuracy test that assert multidim of ndarray gives same results as looping 1d slices
    fs = 1000
    funcs = [lambda sig : compute_spectrum(sig, fs), compute_autocorr]
    sigs = np.random.rand(2, 2, 2, 1, fs)
    sigs2d = sigs.reshape(-1, fs)

    funcs = [
        lambda x : filter_signal(x, fs, 'bandpass', (10, 20), remove_edges=False),
        robust_hilbert,
    ]
    for cfunc in funcs:
        out = cfunc(sigs)
        out2d = out.reshape(-1, out.shape[-1])
        for ind, csig in enumerate(sigs2d):
            out1d = cfunc(csig)
            assert np.all(out1d == out2d[ind])
