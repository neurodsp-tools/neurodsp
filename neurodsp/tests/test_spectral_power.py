"""Test spectral power functions."""

from neurodsp.tests.settings import FS, FREQS_LST, FREQS_ARR, EPS

from neurodsp.spectral.power import *

from neurodsp.sim import sim_oscillation

import numpy as np

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

def test_compute_spectrum_welch(tsig):

    freqs, spectrum = compute_spectrum_welch(tsig, FS, avg_type='mean')
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum_welch(tsig, FS, avg_type='median')
    assert freqs.shape == spectrum.shape

    # Create a sinusoid with 10 periods/cycles.
    n_seconds = 10
    fs = 100
    sig = sim_oscillation(n_seconds, fs, 1, n_cycles=n_seconds, mean=None, variance=None)

    # Use a rectangular window with a width of one period/cycle and no overlap.
    # The specturm should just be a dirac spike at the first frequency.
    window = np.ones(fs)
    _, psd_welch = compute_spectrum(sig, fs, method='welch', nperseg=100, noverlap=0, window=window)

    # Spike at frequency 1.
    assert np.abs(psd_welch[1] - 0.5) < EPS

    # PSD at higher frequencies are essentially zero.
    expected_answer = np.zeros_like(psd_welch[2:])
    assert np.allclose(psd_welch[2:], expected_answer, atol=EPS)

    # No DC component.
    assert np.abs(psd_welch[0]) < EPS

def test_compute_spectrum_wavelet(tsig):

    freqs, spectrum = compute_spectrum_wavelet(tsig, FS, freqs=FREQS_ARR, avg_type='mean')
    assert freqs.shape == spectrum.shape

    freqs, spectrum = compute_spectrum_wavelet(tsig, FS, freqs=FREQS_LST, avg_type='median')
    assert freqs.shape == spectrum.shape

def test_compute_spectrum_medfilt(tsig):

    # Create a sinusoid with 10 periods/cycles.
    n_seconds = 10
    fs = 100
    sig = sim_oscillation(n_seconds, fs, 1, n_cycles=n_seconds, mean=None, variance=None)

    freqs, spectrum = compute_spectrum_medfilt(tsig, FS)
    assert freqs.shape == spectrum.shape

    # Compute raw estimate of psd using fourier transform. Only look at the spectrum up to the Nyquist frequency.
    sig_ft = np.fft.fft(sig)[:len(sig)//2]
    psd = np.abs(sig_ft)**2/(fs * len(sig))

    # The medfilt here should only be taking the median of a window of one sample,
    # so it should agree with our estimate of psd above.
    _, psd_medfilt = compute_spectrum(sig, fs, method='medfilt', filt_len=0.1)

    assert np.allclose(psd, psd_medfilt, atol=EPS)
