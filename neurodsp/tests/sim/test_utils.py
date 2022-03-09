"""Tests for neurodsp.sim.utils."""

import numpy as np

from neurodsp.tests.tutils import check_sim_output
from neurodsp.tests.settings import FS

from neurodsp.sim.utils import *

###################################################################################################
###################################################################################################

def test_rotate_timeseries(tsig):

    out = rotate_timeseries(tsig, FS, 0.25)
    assert out.shape == tsig.shape
    assert not np.array_equal(out, tsig)

def test_rotate_spectrum():

    freqs = np.array([5, 6, 7, 8, 9])
    pows = np.array([1, 2, 3, 4, 5])
    d_exp = 1

    pows_new = rotate_spectrum(freqs, pows, d_exp)
    assert pows.shape == pows_new.shape

def test_modulate_signal(tsig):

    # Check modulation applied by specifying a function name
    msig1 = modulate_signal(tsig, 'sim_oscillation', FS, {'freq' : 1})
    check_sim_output(msig1)

    # Check modulation passing in a 1d array directly
    msig2 = modulate_signal(tsig, tsig)
    check_sim_output(msig2)
