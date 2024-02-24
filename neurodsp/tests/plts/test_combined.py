"""Tests for neurodsp.plts.combined."""

from neurodsp.tests.settings import TEST_PLOTS_PATH, N_SECONDS, FS
from neurodsp.tests.tutils import plot_test

from neurodsp.plts.combined import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_timeseries_and_spectrum(tsig_comb):

    plot_timeseries_and_spectrum(tsig_comb, FS,
                                 save_fig=True, file_path=TEST_PLOTS_PATH,
                                 file_name='test_plot_combined_ts_psd.png')

    # Test customizations
    plot_timeseries_and_spectrum(tsig_comb, FS,
                                 f_range=(3, 50), color='blue',
                                 spectrum_kwargs={'nperseg' : 500},
                                 ts_kwargs={'xlim' : [0, 5]},
                                 psd_kwargs={'lw' : 2},
                                 save_fig=True, file_path=TEST_PLOTS_PATH,
                                 file_name='test_plot_combined_ts_psd2.png')
