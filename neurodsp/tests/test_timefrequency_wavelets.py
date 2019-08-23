"""Tests for timefrequency estimations using wavelets."""

from neurodsp.tests.settings import FS, FREQ1, FREQS_ARR

from neurodsp.timefrequency.wavelets import *

###################################################################################################
###################################################################################################

def test_compute_wavelet_transform(tsig):

    out = compute_wavelet_transform(tsig, FS, FREQS_ARR)

    # Check using a list of n_cycles defintions
    out = compute_wavelet_transform(tsig, FS, FREQS_ARR, n_cycles=[3, 4, 5])

def test_convolve_wavelet(tsig):

    out = convolve_wavelet(tsig, FS, FREQ1)
    out = convolve_wavelet(tsig, FS, FREQ1, norm='amp')
