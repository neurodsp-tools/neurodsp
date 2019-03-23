"""Tests for sim utility functions."""

import numpy as np

from neurodsp.sim.utils import *

###################################################################################################
###################################################################################################

def test_demean():

    dat = np.array([2, 2, 2])
    dat = demean(dat)

    # TODO: Add isclose checking here
    assert True

def test_normalize_variance():

    dat = np.array([2, 2, 2])
    dat = normalize_variance(dat)

    # TODO: Add isclose checking here
    assert True

def test_proportional_sum():
    pass
