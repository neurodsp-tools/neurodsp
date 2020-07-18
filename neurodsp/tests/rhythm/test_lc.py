"""Test lagged coherence code."""

from pytest import warns

import numpy as np

from neurodsp.tests.settings import FS, FREQ1, FREQS_ARR, FREQS_LST

from neurodsp.rhythm.lc import *

###################################################################################################
###################################################################################################

def test_compute_lagged_coherence(tsig):

    lc = compute_lagged_coherence(tsig, FS, FREQS_ARR)
    assert isinstance(lc, float)

    lcs, freqs = compute_lagged_coherence(tsig, FS, FREQS_LST, return_spectrum=True)
    assert sum(np.isnan(lcs)) == 0

    # Check using a list of n_cycles definitions
    lc = compute_lagged_coherence(tsig, FS, FREQS_ARR, n_cycles=[3, 4, 5])

    # Test the warning if can't estimate some values
    with warns(UserWarning):
        compute_lagged_coherence(tsig, 100, np.array([1, 2]), n_cycles=10)

def test_compute_lagged_coherence_2d(tsig2d):

    lc = compute_lagged_coherence(tsig2d, FS, FREQS_ARR)
    assert len(lc) == tsig2d.shape[0]

    lcs, freqs = compute_lagged_coherence(tsig2d, FS, FREQS_ARR, return_spectrum=True)
    assert lcs.shape[0] == tsig2d.shape[0]
    assert freqs.ndim == 1

def test_lagged_coherence_1freq(tsig):

    lc = lagged_coherence_1freq(tsig, FS, FREQ1, n_cycles=3)
    assert isinstance(lc, float)
