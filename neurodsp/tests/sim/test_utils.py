"""Tests for neurodsp.sim.utils."""

import numpy as np

from neurodsp.tests.settings import FS

from neurodsp.sim.utils import *

###################################################################################################
###################################################################################################

def test_rotate_timeseries(tsig):

    out = rotate_timeseries(tsig, FS, 0.25)
    assert out.shape == tsig.shape
    assert not np.array_equal(out, tsig)

def test_rotate_powerlaw():

    freqs = np.array([5, 6, 7, 8, 9])
    pows = np.array([1, 2, 3, 4, 5])
    d_exp = 1

    pows_new = rotate_powerlaw(freqs, pows, d_exp)
    assert pows.shape == pows_new.shape
