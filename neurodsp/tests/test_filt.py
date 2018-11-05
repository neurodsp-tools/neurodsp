"""
test_filt.py
Test filtering functions
"""

import pytest
import numpy as np

from neurodsp.filt import filter_signal
from .util import _load_example_data, _generate_random_sig


def test_bandpass_filter_consistent():
    """
    Confirm consistency in beta bandpass filter results on a neural signal
    """

    # Load data and ground-truth filtered signal
    sig, sig_filt_true = _load_example_data(data_idx=1, filtered=True)

    # filter data
    fs = 1000
    fc = (13, 30)
    sig_filt = filter_signal(sig, fs, 'bandpass', fc=fc, n_cycles=3)

    # Compute difference between current and past filtered signals
    signal_diff = sig_filt[~np.isnan(sig_filt)] - sig_filt_true[~np.isnan(sig_filt_true)]
    assert np.allclose(np.sum(np.abs(signal_diff)), 0, atol=10 ** -5)


def test_edge_nan():
    """
    Confirm that the appropriate amount of edge artifact has been removed for FIR filters
    and that edge artifacts are not removed for IIR filters
    """

    # Apply a 4-8Hz bandpass filter to random noise
    sig = _generate_random_sig()
    sig_filt, kernel = filter_signal(sig, 1000, 'bandpass', fc=(4, 8), return_kernel=True)

    # Check if the correct edge artifacts have been removed
    n_rmv = int(np.ceil(len(kernel) / 2))
    assert all(np.isnan(sig_filt[:n_rmv]))
    assert all(np.isnan(sig_filt[-n_rmv:]))
    assert all(~np.isnan(sig_filt[n_rmv:-n_rmv]))

    # Check that no edge artifacts are removed for IIR filters
    sig_filt = filter_signal(sig, 1000, 'bandpass', fc=(4, 8), iir=True, butterworth_order=3)
    assert all(~np.isnan(sig_filt))


def test_filter_length_error():
    """
    Confirm that the proper error is raised when the filter designed is longer than
    the signal
    """
    n_seconds = 2
    fs = 1000
    sig = np.random.randn(n_seconds * fs)
    with pytest.raises(ValueError) as excinfo:
        sig_filt = filter_signal(sig, fs, 'bandpass', fc=(1, 10))
    assert 'The filter needs to be shortened by decreasing the n_cycles' in str(excinfo.value)


def test_frequency_input_errors():
    """
    Check that errors are properly raised when incorrectly enter frequency information
    """

    # Generate a random signal
    sig = _generate_random_sig()

    # Check that a bandpass filter cannot be completed without proper frequency limits
    with pytest.raises(ValueError):
        sig_filt = filter_signal(sig, 1000, 'bandpass', fc=8)
    with pytest.raises(ValueError):
        sig_filt = filter_signal(sig, 1000, 'bandpass', fc=(8, 4))

    # Check that a bandstop filter cannot be completed without proper frequency limits
    with pytest.raises(ValueError):
        sig_filt = filter_signal(sig, 1000, 'bandstop', fc=58)
    with pytest.raises(ValueError):
        sig_filt = filter_signal(sig, 1000, 'bandstop', fc=(62, 58))

    # Check that a float or partially filled tuple for fc is passable
    sig_filt = filter_signal(sig, 1000, 'lowpass', fc=58)
    sig_filt = filter_signal(sig, 1000, 'lowpass', fc=(0,58))
    sig_filt = filter_signal(sig, 1000, 'highpass', fc=58)
    sig_filt = filter_signal(sig, 1000, 'highpass', fc=(58,1000))

    # Check that frequencies cannot be inverted
    with pytest.raises(ValueError):
        sig_filt = filter_signal(sig, 1000, 'bandpass', fc=(100, 10))


def test_filter_length():
    """
    Check that the output kernel is of the correct length
    """

    # Generate a random signal
    sig = _generate_random_sig()

    # Specify filter length with number of cycles
    fs = 1000
    fc = (4, 8)
    n_cycles = 5
    sig_filt, kernel = filter_signal(sig, fs, 'bandpass', fc=fc,
                                     n_cycles=n_cycles, return_kernel=True)

    # Compute how long the kernel should be
    force_kernel_length = int(np.ceil(fs * n_cycles / fc[0]))
    if force_kernel_length % 2 == 0:
        force_kernel_length = force_kernel_length + 1

    # Check correct length when defining number of cycles
    assert np.allclose(len(kernel), force_kernel_length, atol=.1)

    # Specify filter length with number of seconds
    fs = 1000
    n_seconds = .8
    sig_filt, kernel = filter_signal(sig, fs, 'bandpass', fc=fc,
                                     n_seconds=n_seconds, return_kernel=True)

    # Compute how long the kernel should be
    force_kernel_length = int(np.ceil(fs * n_seconds))
    if force_kernel_length % 2 == 0:
        force_kernel_length = force_kernel_length + 1

    # Check correct length when defining number of seconds
    assert np.allclose(len(kernel), force_kernel_length, atol=.1)
