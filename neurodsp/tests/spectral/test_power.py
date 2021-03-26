"""Tests for neurodsp.spectral.power."""

import numpy as np

from neurodsp.tests.settings import FS, FREQ_SINE, FREQS_LST, FREQS_ARR, EPS

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

def test_compute_spectrum_welch(tsig, tsig_sine):

    freqs, spectrum = compute_spectrum_welch(tsig, FS, avg_type='mean')
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum_welch(tsig, FS, avg_type='median')
    assert freqs.shape == spectrum.shape

    # Use a rectangular window with a width of one period/cycle and no overlap
    #   The spectrum should just be a dirac spike at the first frequency
    window = np.ones(FS)
    _, psd_welch = compute_spectrum(tsig_sine, FS, method='welch',
                                    nperseg=FS, noverlap=0, window=window)

    # Spike at frequency 1
    assert np.abs(psd_welch[FREQ_SINE] - 0.5) < EPS

    # PSD at higher frequencies are essentially zero
    expected_answer = np.zeros_like(psd_welch[FREQ_SINE+1:])
    assert np.allclose(psd_welch[FREQ_SINE+1:], expected_answer, atol=EPS)

    # No DC component or frequencies below the sine frequency
    expected_answer = np.zeros_like(psd_welch[0:FREQ_SINE])
    assert np.allclose(psd_welch[0:FREQ_SINE], expected_answer, atol=EPS)

def test_compute_spectrum_wavelet(tsig):

    freqs, spectrum = compute_spectrum_wavelet(tsig, FS, freqs=FREQS_ARR, avg_type='mean')
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum_wavelet(tsig, FS, freqs=FREQS_LST, avg_type='median')
    assert freqs.shape == spectrum.shape

def test_compute_spectrum_medfilt(tsig, tsig_sine):

    freqs, spectrum = compute_spectrum_medfilt(tsig, FS)
    assert freqs.shape == spectrum.shape

    # Compute raw estimate of psd using fourier transform
    #   Only look at the spectrum up to the Nyquist frequency
    sig_len = len(tsig_sine)
    nyq_freq = sig_len//2
    sig_ft = np.fft.fft(tsig_sine)[:nyq_freq]
    psd = np.abs(sig_ft)**2/(FS * sig_len)

    # The medfilt here should be taking the median of a window with one sample
    #   Therefore, it should match the estimate of psd from above
    _, psd_medfilt = compute_spectrum(tsig_sine, FS, method='medfilt', filt_len=0.1)
    assert np.allclose(psd, psd_medfilt, atol=EPS)
