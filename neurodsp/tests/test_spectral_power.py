"""Test spectral power functions."""

import numpy as np

from neurodsp.spectral.power import *

###################################################################################################
###################################################################################################

def test_compute_spectrum(tsig):

    freqs, spectrum = compute_spectrum(tsig, fs=500, method='welch')
    freqs, spectrum = compute_spectrum(tsig, fs=500, method='wavelet', freqs=[5, 10, 15])
    freqs, spectrum = compute_spectrum(tsig, fs=500, method='medfilt')
    assert True

def test_compute_spectrum_welch(tsig):

    freqs, spectrum = compute_spectrum_welch(tsig, fs=500, avg_type='mean')
    freqs, spectrum = compute_spectrum_welch(tsig, fs=500, avg_type='median')
    assert True

def test_compute_spectrum_wavelet(tsig):

    freqs, spectrum = compute_spectrum_wavelet(tsig, fs=500, freqs=np.array([5, 10, 15]), avg_type='mean')
    freqs, spectrum = compute_spectrum_wavelet(tsig, fs=500, freqs=np.array([5, 10, 15]), avg_type='median')
    assert True

def test_compute_spectrum_medfilt(tsig):

    # NOTE: this test fails uninformatively if fs is 1000 (for siglength 1000). Something to check

    freqs, spectrum = compute_spectrum_medfilt(tsig, fs=500)
    assert True



