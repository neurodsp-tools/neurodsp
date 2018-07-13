"""
test_filt.py
Test filtering functions
"""

import pytest
import numpy as np
import neurodsp
from neurodsp.tests import _load_example_data, _generate_random_x


def test_bandpass_filter_consistent():
    """
    Confirm consistency in beta bandpass filter results on a neural signal
    """
    # Load data and ground-truth filtered signal
    x, x_filt_true = _load_example_data(data_idx=1, filtered=True)

    # filter data
    Fs = 1000
    fc = (13, 30)
    x_filt = neurodsp.filter(x, Fs, 'bandpass', fc=fc, N_cycles=3)

    # Compute difference between current and past filtered signals
    signal_diff = x_filt[~np.isnan(x_filt)] - x_filt_true[~np.isnan(x_filt_true)]
    assert np.allclose(np.sum(np.abs(signal_diff)), 0, atol=10 ** -5)


def test_edge_nan():
    """
    Confirm that the appropriate amount of edge artifact has been removed for FIR filters
    and that edge artifacts are not removed for IIR filters
    """

    # Apply a 4-8Hz bandpass filter to random noise
    x = _generate_random_x()
    x_filt, kernel = neurodsp.filter(x, 1000, 'bandpass', fc=(4, 8), return_kernel=True)

    # Check if the correct edge artifacts have been removed
    N_rmv = int(np.ceil(len(kernel) / 2))
    assert all(np.isnan(x_filt[:N_rmv]))
    assert all(np.isnan(x_filt[-N_rmv:]))
    assert all(~np.isnan(x_filt[N_rmv:-N_rmv]))

    # Check that no edge artifacts are removed for IIR filters
    x_filt = neurodsp.filter(x, 1000, 'bandpass', fc=(4, 8), iir=True, butterworth_order=3)
    assert all(~np.isnan(x_filt))


def test_filter_length_error():
    """
    Confirm that the proper error is raised when the filter designed is longer than
    the signal
    """
    T = 2
    Fs = 1000
    x = np.random.randn(T * Fs)
    with pytest.raises(ValueError) as excinfo:
        x_filt = neurodsp.filt.filter(x, Fs, 'bandpass', fc=(1, 10))
    assert 'The filter needs to be shortened by decreasing the N_cycles' in str(excinfo.value)


def test_frequency_input_errors():
    """
    Check that errors are properly raised when incorrectly enter frequency information
    """

    # Generate a random signal
    x = _generate_random_x()

    # Check that a bandpass filter cannot be completed without proper frequency limits
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'bandpass', fc=8)
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'bandpass', fc=(8, 4))

    # Check that a bandstop filter cannot be completed without proper frequency limits
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'bandstop', fc=58)
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'bandstop', fc=(62, 58))

    # Check that frequencies cannot be inverted
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'lowpass', fc=(100, 10))


def test_filter_length():
    """
    Check that the output kernel is of the correct length
    """

    # Generate a random signal
    x = _generate_random_x()

    # Specify filter length with number of cycles
    Fs = 1000
    fc = (4, 8)
    N_cycles = 5
    x_filt, kernel = neurodsp.filter(x, Fs, 'bandpass', fc=fc,
                                     N_cycles=N_cycles, return_kernel=True)

    # Compute how long the kernel should be
    force_kernel_length = int(np.ceil(Fs * N_cycles / fc[0]))
    if force_kernel_length % 2 == 0:
        force_kernel_length = force_kernel_length + 1

    # Check correct length when defining number of cycles
    assert np.allclose(len(kernel), force_kernel_length, atol=.1)

    # Specify filter length with number of seconds
    Fs = 1000
    N_seconds = .8
    x_filt, kernel = neurodsp.filter(x, Fs, 'bandpass', fc=fc,
                                     N_seconds=N_seconds, return_kernel=True)

    # Compute how long the kernel should be
    force_kernel_length = int(np.ceil(Fs * N_seconds))
    if force_kernel_length % 2 == 0:
        force_kernel_length = force_kernel_length + 1

    # Check correct length when defining number of seconds
    assert np.allclose(len(kernel), force_kernel_length, atol=.1)
