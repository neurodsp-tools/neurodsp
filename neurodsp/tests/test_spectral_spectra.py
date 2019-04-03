"""Test functions in the spectral domain analysis module."""

from neurodsp.spectral.spectra import *

###################################################################################################
###################################################################################################

def test_compute_spectrum(tsig):

    freqs, spectrum = compute_spectrum(tsig, fs=500)
    assert True

def test_compute_spectrum_welch(tsig):

    freqs, spectrum = compute_spectrum_welch(tsig, fs=500, method='mean')
    freqs, spectrum = compute_spectrum_welch(tsig, fs=500, method='median')
    assert True

def test_compute_spectrum_medfilt(tsig):

    # NOTE: this test fails uninformatively if fs is 1000 (for siglength 1000). Something to check

    freqs, spectrum = compute_spectrum_medfilt(tsig, fs=500)
    assert True

## PRIVATE FUNCTIONS
def test_spg_settings():
    pass
