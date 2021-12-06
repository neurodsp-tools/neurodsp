"""Tests for neurodsp.aperiodic.irasa."""

import numpy as np

from neurodsp.sim import sim_combined
from neurodsp.spectral import compute_spectrum, trim_spectrum

from neurodsp.tests.settings import FS, EXP1

from neurodsp.aperiodic.irasa import *

###################################################################################################
###################################################################################################

def test_compute_irasa(tsig_comb):

    # Estimate periodic and aperiodic components with IRASA
    f_range = [1, 30]
    freqs, psd_ap, psd_pe = compute_irasa(tsig_comb, FS, f_range, thresh=0.1, noverlap=int(2*FS))
    assert len(freqs) == len(psd_ap) == len(psd_pe)

    # Compute r-squared for the full model, comparing to a standard power spectrum
    _, powers = trim_spectrum(*compute_spectrum(tsig_comb, FS, nperseg=int(4*FS)), f_range)
    r_sq = np.corrcoef(np.array([powers, psd_ap+psd_pe]))[0][1]
    assert r_sq > .95

def test_fit_irasa(tsig_comb):

    # Estimate periodic and aperiodic components with IRASA & fit aperiodic
    freqs, psd_ap, _ = compute_irasa(tsig_comb, FS, f_range=[1, 30], noverlap=int(2*FS))
    b0, b1 = fit_irasa(freqs, psd_ap)

    assert round(b1) == EXP1
    assert np.abs(b0 - np.log10((psd_ap)[0])) < 1

def test_fit_func():

    freqs = np.arange(30)
    intercept = -2
    slope = -2

    fit = fit_func(freqs, intercept, slope)
    assert (fit == slope * freqs + intercept).all()
