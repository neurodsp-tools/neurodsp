"""Test spectral plots."""

import numpy as np

from neurodsp.spectral import compute_spectral_hist
from neurodsp.tests.util import plot_test

from neurodsp.plts.spectral import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_spectral_hist():

    # Generate random signal & compute spectral histogram
    sig = np.random.randn(1000)
    freqs, power_bins, spect_hist = compute_spectral_hist(sig, fs=1000)

    # Test plotting function runs without error
    plot_spectral_hist(freqs, power_bins, spect_hist)
