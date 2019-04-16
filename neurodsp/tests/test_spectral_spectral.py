"""Test functions in the spectral domain analysis module."""

from neurodsp.spectral.spectral import *

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

    freqs, spectrum = compute_spectrum_wavelet(tsig, fs=500, freqs=[5, 10, 15], avg_type='mean')
    freqs, spectrum = compute_spectrum_wavelet(tsig, fs=500, freqs=[5, 10, 15], avg_type='median')
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

    freqs, bins, spectral_hist = compute_spectral_hist(tsig, fs=500)
    assert True

## PRIVATE FUNCTIONS
def test_spg_settings():
    pass
