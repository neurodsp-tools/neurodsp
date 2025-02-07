"""Tests for neurodsp.plts.aperiodic."""

from neurodsp.aperiodic.autocorr import compute_autocorr

from neurodsp.tests.settings import TEST_PLOTS_PATH, FS
from neurodsp.tests.tutils import plot_test

from neurodsp.plts.aperiodic import *

###################################################################################################
###################################################################################################

def tests_plot_autocorr(tsig, tsig_comb):

    times1, acs1 = compute_autocorr(tsig, max_lag=150)
    times2, acs2 = compute_autocorr(tsig_comb, max_lag=150)

    plot_autocorr(times1, acs1,
                  save_fig=True, file_path=TEST_PLOTS_PATH,
                  file_name='test_plot_autocorr-1.png')

    plot_autocorr([times1, times2], [acs1, acs2],
                  labels=['first', 'second'], colors=['k', 'r'],
                  save_fig=True, file_path=TEST_PLOTS_PATH,
                  file_name='test_plot_autocorr-2.png')
