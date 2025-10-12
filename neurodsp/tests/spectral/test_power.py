"""Tests for neurodsp.spectral.power."""

from pytest import raises

import numpy as np

from neurodsp.tests.tsettings import FS, FREQ_SINE, FREQS_LST, FREQS_ARR, EPS

from neurodsp.spectral.power import *
from neurodsp.spectral.power import _spectrum_input_checks

###################################################################################################
###################################################################################################

def test_compute_spectrum(tsig):

    freqs, spectrum = compute_spectrum(tsig, FS, method='welch')
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum(tsig, FS, method='wavelet', freqs=FREQS_ARR)
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum(tsig, FS, method='medfilt')
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum(tsig, FS, method='multitaper')
    assert freqs.shape == spectrum.shape

def test_spectrum_input_checks():

    # Test consistent examples
    _spectrum_input_checks('welch', {'nperseg' : 500, 'noverlap' : 250})
    _spectrum_input_checks('medfilt', {'filt_len' : 500})

    # Test inconsistent examples
    with raises(AssertionError):
        _spectrum_input_checks('welch', {'filt_len' : 500})
        _spectrum_input_checks('welch', {'nonsense' : 500})

def test_compute_spectrum_2d(tsig2d):

    freqs, spectrum = compute_spectrum(tsig2d, FS, method='welch')
    assert freqs.shape[-1] == spectrum.shape[-1]
    assert spectrum.ndim == 2

    freqs, spectrum = compute_spectrum(tsig2d, FS, method='wavelet', freqs=FREQS_ARR)
    assert freqs.shape[-1] == spectrum.shape[-1]
    assert spectrum.ndim == 2

    freqs, spectrum = compute_spectrum(tsig2d, FS, method='medfilt')
    assert freqs.shape[-1] == spectrum.shape[-1]
    assert spectrum.ndim == 2

    freqs, spectrum = compute_spectrum(tsig2d, FS, method='multitaper')
    assert freqs.shape[-1] == spectrum.shape[-1]
    assert spectrum.ndim == 2

def test_compute_spectrum_fft(tsig, tsig_sine):

    freqs1, spectrum1 = compute_spectrum_fft(tsig, FS)
    assert freqs1.shape == spectrum1.shape

    # Test applying a window function
    freqs2, spectrum2 = compute_spectrum_fft(tsig, FS, window='hann')
    assert freqs2.shape == spectrum2.shape

    # Test padding signal
    freqs3, spectrum3 = compute_spectrum_fft(tsig, FS, nfft=1.5*len(tsig))
    assert freqs3.shape == spectrum3.shape
    assert freqs2.shape != freqs3.shape

def test_compute_spectrum_welch(tsig, tsig_sine):

    freqs, spectrum = compute_spectrum_welch(tsig, FS, avg_type='mean')
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum_welch(tsig, FS, avg_type='median')
    assert freqs.shape == spectrum.shape

    # Use a rectangular window with a width of one period/cycle and no overlap
    #   The spectrum should just be a dirac spike at the first frequency
    window = np.ones(FS)
    _, psd_welch = compute_spectrum_welch(tsig_sine, FS, nperseg=FS, noverlap=0, window=window)

    # Spike at frequency 1
    assert np.abs(psd_welch[FREQ_SINE] - 0.5) < EPS

    # PSD at higher frequencies are essentially zero
    expected_answer = np.zeros_like(psd_welch[FREQ_SINE+1:])
    assert np.allclose(psd_welch[FREQ_SINE+1:], expected_answer, atol=EPS)

    # No DC component or frequencies below the sine frequency
    expected_answer = np.zeros_like(psd_welch[0:FREQ_SINE])
    assert np.allclose(psd_welch[0:FREQ_SINE], expected_answer, atol=EPS)

    # Test zero padding
    freqs, spectrum = compute_spectrum_welch(
        np.tile(tsig, (2, 1)), FS, nperseg=100, noverlap=0, nfft=1000, f_range=(1, 200)
    )
    assert np.all(spectrum[0] == spectrum[1])

def test_compute_spectrum_wavelet(tsig):

    freqs, spectrum = compute_spectrum_wavelet(tsig, FS, freqs=FREQS_ARR, avg_type='mean')
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum_wavelet(tsig, FS, freqs=FREQS_LST, avg_type='median')
    assert freqs.shape == spectrum.shape

def test_compute_spectrum_medfilt(tsig, tsig_sine):

    freqs, spectrum = compute_spectrum_medfilt(tsig, FS)
    assert freqs.shape == spectrum.shape

    # Compute raw estimate of psd using FFT
    sig_len = len(tsig_sine)
    psd = np.abs(np.fft.rfft(tsig_sine))**2 / (FS * sig_len)

    # The medfilt here should be taking the median of a window with one sample
    #   Therefore, it should match the estimate of psd from above
    _, psd_medfilt = compute_spectrum_medfilt(tsig_sine, FS, filt_len=0.1)
    assert np.allclose(psd, psd_medfilt, atol=EPS)

def test_compute_spectrum_multitaper(tsig_sine, tsig2d):
    # Shape test: 1D input
    freqs, spectrum = compute_spectrum_multitaper(tsig_sine, FS)
    assert freqs.shape == spectrum.shape

    # Shape test: 2D input
    freqs_2d, spectrum_2d = compute_spectrum_multitaper(tsig2d, FS)
    assert spectrum_2d.ndim == 2
    assert spectrum_2d.shape[0] == tsig2d.shape[0]
    assert spectrum_2d.shape[1] == len(freqs_2d)

    # Accuracy test: peak at sine frequency
    idx_freq_sine = np.argmin(np.abs(freqs - FREQ_SINE))
    idx_peak = np.argmax(spectrum)
    assert idx_freq_sine == idx_peak
