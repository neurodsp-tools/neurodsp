"""Tests for outlier related utility functions."""

import numpy as np
from numpy.testing import assert_equal

from neurodsp.utils.outliers import *

###################################################################################################
###################################################################################################

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
    arr_nans = np.array([True, True, False, False, False, True])

    arr_restored = restore_nans(arr_no_nans, arr_nans)
    assert_equal(arr_restored, np.array([np.NaN, np.NaN, 1, 2, 3, np.NaN]))

def test_discard_outliers():

    dat = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 10],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]])
    new_dat = discard_outliers(dat, 0.10)

    assert_equal(new_dat, np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1]]))
