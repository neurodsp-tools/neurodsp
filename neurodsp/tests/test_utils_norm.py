"""Tests for normalization related utility functions."""

import numpy as np

from neurodsp.utils.norm import *

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
