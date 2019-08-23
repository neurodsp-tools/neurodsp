"""Test spectral variance functions."""

from neurodsp.tests.settings import FS

from neurodsp.spectral.variance import *

###################################################################################################
###################################################################################################

def test_compute_scv(tsig):

    freqs, spect_cv = compute_scv(tsig, FS)

def test_compute_scv_rs(tsig):

    freqs, t_inds, spect_cv = compute_scv_rs(tsig, FS, method='bootstrap')
    freqs, t_inds, spect_cv = compute_scv_rs(tsig, FS, method='rolling')

def test_compute_spectral_hist(tsig):

    freqs, bins, spectral_hist = compute_spectral_hist(tsig, FS)
