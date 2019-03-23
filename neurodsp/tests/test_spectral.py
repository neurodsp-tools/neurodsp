"""Test functions in the spectral domain analysis module."""

from neurodsp.spectral import *

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

def test_compute_scv(tsig):

    freqs, spect_cv = compute_scv(tsig, fs=500)
    assert True

def test_compute_scv_rs(tsig):

    freqs, t_inds, spect_cv = compute_scv_rs(tsig, fs=500, method='bootstrap')
    freqs, t_inds, spect_cv = compute_scv_rs(tsig, fs=500, method='rolling')
    assert True

def test_compute_spectral_hist(tsig):

    freqs, bins, spect_hist = compute_spectral_hist(tsig, fs=500)
    assert True

def test_morlet_transform(tsig):

    out = morlet_transform(tsig, [5, 10, 15], fs=500)
    assert True

def test_morlet_convolve(tsig):

    out = morlet_convolve(tsig, 10, fs=500)
    assert True

def test_rotate_powerlaw():
    pass

def test_trim_spectrum():
    pass

## PRIVATE FUNCTIONS

def test_discard_outliers():
    pass

def test_spg_settings():
    pass


