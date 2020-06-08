"""Test time series plots."""

from pytest import raises

from neurodsp.tests.utils import plot_test

from neurodsp.plts.time_series import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_time_series(tsig):

    times = np.arange(0, len(tsig), 1)

    # Run single time series plot
    plot_time_series(times, tsig)

    # Run multi time series plot, with colors & labels
    plot_time_series(times, [tsig, tsig[::-1]],
                     labels=['signal', 'signal reversed'], colors=['k', 'r'])

@plot_test
def test_plot_instantaneous_measure(tsig):

    times = np.arange(0, len(tsig), 1)

    plot_instantaneous_measure(times, tsig, 'phase')
    plot_instantaneous_measure(times, tsig, 'amplitude')
    plot_instantaneous_measure(times, tsig, 'frequency')

    # Check the error for bad measure
    with raises(ValueError):
        plot_instantaneous_measure(times, tsig, 'BAD')

@plot_test
def test_plot_bursts(tsig):

    times = np.arange(0, len(tsig), 1)
    bursts = np.array([True] * len(tsig))

    plot_bursts(times, tsig, bursts)
