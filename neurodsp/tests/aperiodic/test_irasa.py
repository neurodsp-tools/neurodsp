"""Tests for IRASA functions."""

import numpy as np

from neurodsp.tests.settings import FS, N_SECONDS_LONG

from neurodsp.sim import sim_combined
from neurodsp.spectral import compute_spectrum, trim_spectrum
from neurodsp.aperiodic.irasa import irasa, fit_irasa, fit_func

###################################################################################################
###################################################################################################


def test_irasa():

    # Simulate a signal
    sim_components = {'sim_powerlaw': {'exponent' : -2},
                      'sim_oscillation': {'freq' : 10}}
    sig = sim_combined(n_seconds=N_SECONDS_LONG, fs=FS, components=sim_components)
    _, powers = trim_spectrum(*compute_spectrum(sig, FS, nperseg=int(4*FS)), [1, 30])

    # Estimate periodic and aperiodic components with IRASA
    freqs, psd_ap, psd_pe = irasa(sig, FS, noverlap=int(2*FS))

    # Compute r-squared for the full model
    r_sq = np.corrcoef(np.array([powers, psd_ap+psd_pe]))[0][1]

    assert r_sq > .95
    assert len(freqs) == len(psd_ap) == len(psd_pe)


def test_fit_irasa():

    knee = -1

    # Simulate a signal
    sim_components = {'sim_powerlaw': {'exponent' : knee},
                      'sim_oscillation': {'freq' : 10}}
    sig = sim_combined(n_seconds=N_SECONDS_LONG, fs=FS, components=sim_components)

    # Estimate periodic and aperiodic components with IRASA
    freqs, psd_ap, _ = irasa(sig, FS, noverlap=int(2*FS))

    # Get aperiodic coefficients
    b0, b1 = fit_irasa(freqs, psd_ap)

    assert round(b1) == knee
    assert np.abs(b0 - np.log((psd_ap)[0])) < 1


def test_fit_func():

    freqs = np.arange(30)
    intercept = -2
    slope = -2

    fit = fit_func(freqs, intercept, slope)
    assert (fit == slope * freqs + intercept).all()
