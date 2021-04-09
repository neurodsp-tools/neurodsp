"""Tests for neurodsp.plts.timefrequency."""

import numpy as np

from neurodsp.utils import create_times
from neurodsp.timefrequency.wavelets import compute_wavelet_transform

from neurodsp.tests.settings import TEST_PLOTS_PATH, N_SECONDS, FS
from neurodsp.tests.tutils import plot_test

from neurodsp.plts.timefrequency import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_timefrequency(tsig_comb):

    times = create_times(N_SECONDS, FS)
    freqs = np.array([5, 10, 15, 20, 25])
    mwt = compute_wavelet_transform(tsig_comb, FS, freqs)

    plot_timefrequency(times, freqs, mwt,
                       save_fig=True, file_path=TEST_PLOTS_PATH,
                       file_name='test_plot_timefrequency1.png')

    plot_timefrequency(times, freqs, mwt, x_ticks=[2.5, 7.5], y_ticks=[5, 15],
                       save_fig=True, file_path=TEST_PLOTS_PATH,
                       file_name='test_plot_timefrequency2.png')
