"""Test spectral plots."""

import numpy as np

from neurodsp.spectral.variance import compute_spectral_hist
from neurodsp.tests.utils import plot_test

from neurodsp.plts.spectral import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_power_spectra():

    freqs, powers = np.array([1, 2, 3, 4]), np.array([10, 20, 30, 40])
    plot_power_spectra(freqs, powers)
    plot_power_spectra([freqs, freqs], [powers, powers])

@plot_test
def test_plot_scv():

    freqs, scv = np.array([1, 2, 3, 4]), np.array([10, 20, 30, 40])
    plot_scv(freqs, scv)

@plot_test
def test_plot_scv_rs_lines():

    freqs, scv_rs = np.array([1, 2, 3]), np.array([[2, 3, 4], [2, 3, 4], [3, 4, 5]])
    plot_scv_rs_lines(freqs, scv_rs)

@plot_test
def test_plot_scv_rs_matrix():

    freqs, times = np.array([1, 2, 3]), np.array([1, 2, 3])
    scv_rs = np.array([[2, 3, 4], [2, 3, 4], [3, 4, 5]])
    plot_scv_rs_matrix(freqs, times, scv_rs)

@plot_test
def test_plot_spectral_hist(tsig):

    freqs, power_bins, spectral_hist = compute_spectral_hist(tsig, fs=1000)
    plot_spectral_hist(freqs, power_bins, spectral_hist)
