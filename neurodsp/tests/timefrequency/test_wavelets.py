"""Tests for time-frequency estimations using wavelets."""

from neurodsp.tests.settings import FS, FREQ1, FREQS_ARR

from neurodsp.timefrequency.wavelets import *

###################################################################################################
###################################################################################################

def test_compute_wavelet_transform(tsig):

    out = compute_wavelet_transform(tsig, FS, FREQS_ARR)
    assert out.ndim == 2

    # Check using a list of n_cycles definitions
    out = compute_wavelet_transform(tsig, FS, FREQS_ARR, n_cycles=[3, 4, 5])

def test_compute_wavelet_transform_2d(tsig2d):

    out = compute_wavelet_transform(tsig2d, FS, FREQS_ARR)
    assert out.ndim == 3

def test_convolve_wavelet(tsig):

    out = convolve_wavelet(tsig, FS, FREQ1)
    out = convolve_wavelet(tsig, FS, FREQ1, norm='amp')

def test_convolve_wavelet_2d(tsig2d):

    out = convolve_wavelet(tsig2d, FS, FREQ1)
    assert out.shape == tsig2d.shape
