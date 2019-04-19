"""Tests for utility functions."""

import numpy as np
from numpy.testing import assert_equal

from neurodsp.utils.utils import *

###################################################################################################
###################################################################################################

def test_demean():

    d1 = np.array([1, 2, 3])
    d2 = np.array([0, 1, 2, 3, 0])

    # Check default - demean to 0
    out1 = demean(d1)
    assert np.isclose(out1.mean(), 0.)

    # Check demeaning and adding specific mean
    out2 = demean(d1, mean=1.)
    assert np.isclose(out2.mean(), 1.)

    # Check dealing with zero entries
    out3 = demean(d2)
    assert np.isclose(out3[np.nonzero(out3)].mean(), 0)

    # Check turning of non-zero selection
    out3 = demean(d2, mean=1, select_nonzero = False)
    assert np.isclose(out3.mean(), 1)

def test_normalize_variance():

    d1 = np.array([1, 2, 3])
    d2 = np.array([0, 1, 2, 3, 0])

    # Check default - normalize variance to 1
    out1 = normalize_variance(d1)
    np.isclose(out1.var(), 1.)

    # Check normalizing and add specific variance
    out2 = normalize_variance(d1, 2.)
    np.isclose(out2.var(), 2.)

def test_remove_nans():

    # Test with equal # of NaNs on either edge
    arr = np.array([np.NaN, np.NaN, 1, 2, 3, np.NaN, np.NaN])
    arr_no_nans, arr_nans = remove_nans(arr)
    assert_equal(arr_no_nans, np.array([1, 2, 3]))
    assert_equal(arr_nans, np.array([True, True, False, False, False, True, True]))

    # Test with different # of NaNs on either edge
    arr = np.array([np.NaN, np.NaN, 1, 2, 3, 4, np.NaN,])
    arr_no_nans, arr_nans = remove_nans(arr)
    assert_equal(arr_no_nans, np.array([1, 2, 3, 4]))
    assert_equal(arr_nans, np.array([True, True, False, False, False, False, True]))

def test_restore_nans():

    arr_no_nans = np.array([1, 2, 3])
    arr_nans =  np.array([True, True, False, False, False, True])

    arr_restored = restore_nans(arr_no_nans, arr_nans)
    assert_equal(arr_restored, np.array([np.NaN, np.NaN, 1, 2, 3, np.NaN]))

def test_discard_outliers():
    pass
