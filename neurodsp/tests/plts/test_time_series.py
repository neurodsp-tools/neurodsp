"""Tests for neurodsp.plts.time_series."""

from pytest import raises
import numpy as np

from neurodsp.utils import create_times
from neurodsp.burst import detect_bursts_dual_threshold
from neurodsp.timefrequency import amp_by_time, phase_by_time, freq_by_time

from neurodsp.tests.settings import TEST_PLOTS_PATH, N_SECONDS, FS, F_RANGE
from neurodsp.tests.tutils import plot_test

from neurodsp.plts.time_series import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_time_series(tsig_comb, tsig_burst):

    times = create_times(N_SECONDS, FS)

    # Check single time series plot
    plot_time_series(times, tsig_comb,
                     save_fig=True, file_path=TEST_PLOTS_PATH,
                     file_name='test_plot_time_series-1.png')

    # Check multi time series plot, labels
    plot_time_series(times, [tsig_comb, tsig_comb[::-1]],
                     labels=['signal', 'signal reversed'],
                     save_fig=True, file_path=TEST_PLOTS_PATH,
                     file_name='test_plot_time_series-2.png')

    # Test 2D arrays
    plot_time_series(times, np.array([tsig_comb, tsig_burst]),
                     save_fig=True, file_path=TEST_PLOTS_PATH,
                     file_name='test_plot_time_series-2arr.png')

    # Test none input for times
    plot_time_series(None, np.array([tsig_comb, tsig_burst]),
                     save_fig=True, file_path=TEST_PLOTS_PATH,
                     file_name='test_plot_time_series_notimes.png')

@plot_test
def test_plot_instantaneous_measure(tsig_comb):

    times = create_times(N_SECONDS, FS)

    plot_instantaneous_measure(times, amp_by_time(tsig_comb, FS, F_RANGE), 'amplitude',
                               save_fig=True, file_path=TEST_PLOTS_PATH,
                               file_name='test_plot_instantaneous_measure_amplitude.png')

    plot_instantaneous_measure(times, phase_by_time(tsig_comb, FS, F_RANGE), 'phase',
                               save_fig=True, file_path=TEST_PLOTS_PATH,
                               file_name='test_plot_instantaneous_measure_phase.png')

    plot_instantaneous_measure(times, freq_by_time(tsig_comb, FS, F_RANGE), 'frequency',
                               save_fig=True, file_path=TEST_PLOTS_PATH,
                               file_name='test_plot_instantaneous_measure_frequency.png')

    # Check the error for bad measure
    with raises(ValueError):
        plot_instantaneous_measure(times, tsig_comb, 'BAD')

@plot_test
def test_plot_bursts(tsig_burst):

    times = create_times(N_SECONDS, FS)
    bursts = detect_bursts_dual_threshold(tsig_burst, FS, (0.75, 1.5), F_RANGE)

    plot_bursts(times, tsig_burst, bursts,
                save_fig=True, file_path=TEST_PLOTS_PATH,
                file_name='test_plot_bursts.png')
