"""Tests for neurodsp.utils.outliers."""

import numpy as np
from numpy.testing import assert_equal

from neurodsp.utils.outliers import *

###################################################################################################
###################################################################################################

def test_remove_nans():

    # Test with an equal number of NaNs on either edge
    arr = np.array([np.nan, np.nan, 1, 2, 3, np.nan, np.nan])
    arr_no_nans, arr_nans = remove_nans(arr)
    assert_equal(arr_no_nans, np.array([1, 2, 3]))
    assert_equal(arr_nans, np.array([True, True, False, False, False, True, True]))

    # Test with a different number of NaNs on either edge
    arr = np.array([np.nan, np.nan, 1, 2, 3, 4, np.nan,])
    arr_no_nans, arr_nans = remove_nans(arr)
    assert_equal(arr_no_nans, np.array([1, 2, 3, 4]))
    assert_equal(arr_nans, np.array([True, True, False, False, False, False, True]))

def test_restore_nans():

    arr_no_nans = np.array([1, 2, 3])
    arr_nans = np.array([True, True, False, False, False, True])

    arr_restored = restore_nans(arr_no_nans, arr_nans)
    assert_equal(arr_restored, np.array([np.nan, np.nan, 1, 2, 3, np.nan]))

def test_discard_outliers():

    dat = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 10],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 10]])
    new_dat = discard_outliers(dat, 0.10)

    assert_equal(new_dat, np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1]]))

def test_discard_outliers_zero():
    """Zero removal preserves all windows and the original number of axes."""
    for shape in [(2, 4), (1, 4), (1, 2, 4), (2, 1, 4)]:
        data = np.arange(np.prod(shape), dtype=float).reshape(shape)
        result = discard_outliers(data, 0)
        assert_equal(result, data)
        assert result.shape == shape


def test_compute_scv_zero_outliers(tsig):
    """Explicitly disabling removal agrees with the untrimmed signal analysis."""
    from neurodsp.spectral import compute_scv

    freqs, expected = compute_scv(tsig, 500)
    actual_freqs, actual = compute_scv(tsig, 500, outlier_pct=0)
    assert_equal(actual_freqs, freqs)
    assert_equal(actual, expected)
