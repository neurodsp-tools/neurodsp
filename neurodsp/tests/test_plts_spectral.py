"""Test spectral plots."""

import numpy as np

from neurodsp.spectral import spectral_hist
from neurodsp.plts.spectral import plot_spectral_hist
from .util import plot_test

###################################################################################################
###################################################################################################

@plot_test
def test_plot_spectral_hist():

    # Generate random signal
    sig = np.random.randn(2000)
    fs = 1000

    # Compute spectral histogram
    freqs, power_bins, spect_hist = spectral_hist(sig, fs)

    # Test plotting function runs without error
    plot_spectral_hist(freqs, power_bins, spect_hist)
