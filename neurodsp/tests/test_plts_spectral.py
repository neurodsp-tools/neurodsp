"""Test spectral plots."""

from neurodsp.spectral.spectral import compute_spectral_hist
from neurodsp.tests.utils import plot_test

from neurodsp.plts.spectral import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_spectral_hist(tsig):

    freqs, power_bins, spectral_hist = compute_spectral_hist(tsig, fs=1000)
    plot_spectral_hist(freqs, power_bins, spectral_hist)
