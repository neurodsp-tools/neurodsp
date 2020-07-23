"""Tests for normalization related utility functions."""

import numpy as np

from neurodsp.utils.norm import *

###################################################################################################
###################################################################################################

def test_normalize_sig():

    sig1 = np.array([1, 2, 3, 4, 5, 6])
    sig2 = np.array([0, 1, 0, -1, 0, 1, 0, -1])

    new_mean = 2
    new_variance = 2

    for sig in [sig1, sig2]:
        out = normalize_sig(sig, new_mean, new_variance)
        assert np.isclose(out.mean(), new_mean)
        assert np.isclose(out.var(), new_variance)

def test_demean():

    d1 = np.array([1, 2, 3])
    d2 = np.array([0, 1, 2, 3, 0])

    # Check default - demean to 0
    out1 = demean(d1)
    assert np.isclose(out1.mean(), 0.)

    # Check demeaning and adding specific mean
    new_mean = 1.
    out2 = demean(d1, mean=new_mean)
    assert np.isclose(out2.mean(), new_mean)

def test_normalize_variance():

    d1 = np.array([1, 2, 3])
    d2 = np.array([0, 1, 2, 3, 0])

    # Check default - normalize variance to 1
    out1 = normalize_variance(d1)
    np.isclose(out1.var(), 1.)

    # Check normalizing and add specific variance
    out2 = normalize_variance(d1, 2.)
    np.isclose(out2.var(), 2.)
