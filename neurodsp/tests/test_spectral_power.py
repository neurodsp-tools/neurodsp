"""Test spectral power functions."""

from neurodsp.tests.settings import FS, FREQS_LST, FREQS_ARR

from neurodsp.spectral.power import *

###################################################################################################
###################################################################################################

def test_compute_spectrum(tsig):

    freqs, spectrum = compute_spectrum(tsig, FS, method='welch')
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum(tsig, FS, method='wavelet', freqs=FREQS_ARR)
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum(tsig, FS, method='medfilt')
    assert freqs.shape == spectrum.shape

def test_compute_spectrum_welch(tsig):

    freqs, spectrum = compute_spectrum_welch(tsig, FS, avg_type='mean')
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum_welch(tsig, FS, avg_type='median')
    assert freqs.shape == spectrum.shape

def test_compute_spectrum_wavelet(tsig):

    freqs, spectrum = compute_spectrum_wavelet(tsig, FS, freqs=FREQS_ARR, avg_type='mean')
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum_wavelet(tsig, FS, freqs=FREQS_LST, avg_type='median')
    assert freqs.shape == spectrum.shape

def test_compute_spectrum_medfilt(tsig):

    freqs, spectrum = compute_spectrum_medfilt(tsig, FS)
    assert freqs.shape == spectrum.shape
