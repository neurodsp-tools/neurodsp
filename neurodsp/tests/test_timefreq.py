"""Test functions for the time-frequency analysis module."""

import os

import numpy as np

from neurodsp.timefrequency import *
from neurodsp.timefrequency import _hilbert_ignore_nan

from .util import load_example_data

###################################################################################################
###################################################################################################

def test_hilbert_ignore_nan():
    """Check that time-resolved timefrequency functions do not return all NaN
    if one of the elements in the input array is NaN.
    Do this by replacing edge artifacts with NaN for a lowpass filter
    """

    # Generate a signal with NaNs
    fs, n_points, n_nans = 100, 1000, 10
    sig = np.random.randn(n_points)
    sig[0:n_nans] = np.nan

    # Check has correct number of nans (not all nan), without increase_n
    hilb_sig = _hilbert_ignore_nan(sig)
    assert sum(np.isnan(hilb_sig)) == n_nans

    # Check has correct number of nans (not all nan), with increase_n
    hilb_sig = _hilbert_ignore_nan(sig, True)
    assert sum(np.isnan(hilb_sig)) == n_nans

def test_timefreq_consistent():
    """Confirm consistency in estimation of instantaneous phase, amp, and frequency
    with computations in previous versions
    """

    # Load data
    data_idx = 1
    sig = load_example_data(data_idx=data_idx)
    fs = 1000
    f_range = (13, 30)

    # Load ground truth phase time series
    pha_true = np.load(os.path.dirname(neurodsp.__file__) +
                       '/tests/data/sample_data_' + str(data_idx) + '_pha.npy')
    # Load ground truth amplitude time series
    amp_true = np.load(os.path.dirname(neurodsp.__file__) +
                       '/tests/data/sample_data_' + str(data_idx) + '_amp.npy')
    # Load ground truth frequency time series
    i_f_true = np.load(os.path.dirname(neurodsp.__file__) +
                       '/tests/data/sample_data_' + str(data_idx) + '_i_f.npy')

    # Compute phase time series
    pha = phase_by_time(sig, fs, f_range)
    # Compute amplitude time series
    amp = amp_by_time(sig, fs, f_range)
    # Compute frequency time series
    i_f = freq_by_time(sig, fs, f_range)

    # Compute difference between current and past signals
    assert np.allclose(
        np.sum(np.abs(pha[~np.isnan(pha)] - pha_true[~np.isnan(pha)])), 0, atol=10 ** -5)
    assert np.allclose(
        np.sum(np.abs(amp[~np.isnan(amp)] - amp_true[~np.isnan(amp)])), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(i_f[~np.isnan(i_f)] - i_f_true[~np.isnan(i_f_true)])),
                       0, atol=10 ** -5)
