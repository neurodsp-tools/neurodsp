"""Test the utility function from spectral."""

import numpy as np
from numpy.testing import assert_equal

from neurodsp.spectral.utils import *

###################################################################################################
###################################################################################################

def test_trim_spectrum():

    freqs = np.array([5, 6, 7, 8, 9])
    pows = np.array([1, 2, 3, 4, 5])

    freqs_new, pows_new = trim_spectrum(freqs, pows, [6, 8])
    assert_equal(freqs_new, np.array([6, 7, 8]))
    assert_equal(pows_new, np.array([2, 3, 4]))

def test_rotate_powerlaw():

    freqs = np.array([5, 6, 7, 8, 9])
    pows = np.array([1, 2, 3, 4, 5])
    d_exp = 1

    pows_new = rotate_powerlaw(freqs, pows, d_exp)
    assert pows.shape == pows_new.shape
