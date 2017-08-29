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
    f_lo = 13
    f_hi = 30
    x_filt = neurodsp.filter(x, Fs, 'bandpass', f_lo=f_lo, f_hi=f_hi, N_cycles=3)

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
    x_filt, kernel = neurodsp.filter(x, 1000, 'bandpass', f_lo=4, f_hi=8, return_kernel=True)

    # Check if the correct edge artifacts have been removed
    N_rmv = int(np.ceil(len(kernel) / 2))
    assert all(np.isnan(x_filt[:N_rmv]))
    assert all(np.isnan(x_filt[-N_rmv:]))
    assert all(~np.isnan(x_filt[N_rmv:-N_rmv]))

    # Check that no edge artifacts are removed for IIR filters
    x_filt = neurodsp.filter(x, 1000, 'bandpass', f_lo=4, f_hi=8, iir=True, butterworth_order=3)
    assert all(~np.isnan(x_filt))


def test_frequency_input_errors():
    """
    Check that errors are properly raised when incorrectly enter frequency information
    """

    # Generate a random signal
    x = _generate_random_x()

    # Check that a bandpass filter cannot be completed without proper frequency limits
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'bandpass', f_lo=4)
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'bandpass', f_hi=8)
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'bandpass', f_lo=8, f_hi=4)

    # Check that a bandstop filter cannot be completed without proper frequency limits
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'bandstop', f_lo=58)
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'bandstop', f_hi=62)
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'bandstop', f_lo=62, f_hi=58)

    # Check that a lowpass filter cannot be completed without proper frequency limits
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'lowpass', f_hi=100)
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'lowpass', f_lo=10, f_hi=100)

    # Check that a highpass filter cannot be completed without proper frequency limits
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'highpass', f_lo=100)
    with pytest.raises(ValueError):
        x_filt = neurodsp.filter(x, 1000, 'highpass', f_lo=10, f_hi=100)


def test_filter_length():
    """
    Check that the output kernel is of the correct length
    """

    # Generate a random signal
    x = _generate_random_x()

    # Specify filter length with number of cycles
    Fs = 1000
    f_lo = 4
    f_hi = 8
    N_cycles = 5
    x_filt, kernel = neurodsp.filter(x, Fs, 'bandpass', f_lo=f_lo,
                                     f_hi=f_hi, N_cycles=N_cycles, return_kernel=True)

    # Compute how long the kernel should be
    force_kernel_length = int(np.ceil(Fs * N_cycles / f_lo))
    if force_kernel_length % 2 == 0:
        force_kernel_length = force_kernel_length + 1

    # Check correct length when defining number of cycles
    assert np.allclose(len(kernel), force_kernel_length, atol=.1)

    # Specify filter length with number of seconds
    Fs = 1000
    f_lo = 4
    f_hi = 8
    N_seconds = .8
    x_filt, kernel = neurodsp.filter(x, Fs, 'bandpass', f_lo=f_lo,
                                     f_hi=f_hi, N_seconds=N_seconds, return_kernel=True)

    # Compute how long the kernel should be
    force_kernel_length = int(np.ceil(Fs * N_seconds))
    if force_kernel_length % 2 == 0:
        force_kernel_length = force_kernel_length + 1

    # Check correct length when defining number of seconds
    assert np.allclose(len(kernel), force_kernel_length, atol=.1)
