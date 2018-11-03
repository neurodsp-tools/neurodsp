"""
test_timefreq.py
Test functions in the time-frequency analysis module
"""

import numpy as np
import os

import neurodsp
from neurodsp.filt import filter_signal
from neurodsp.timefrequency import amp_by_time, phase_by_time, freq_by_time
from .util import _load_example_data


def test_timefreq_consistent():
    """
    Confirm consistency in estimation of instantaneous phase, amp, and frequency
    with computations in previous versions
    """
    # Load data
    data_idx = 1
    sig = _load_example_data(data_idx=data_idx)
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
    assert np.allclose(np.sum(np.abs(pha[~np.isnan(pha)] - pha_true[~np.isnan(pha)])), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(amp[~np.isnan(amp)] - amp_true[~np.isnan(amp)])), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(i_f[~np.isnan(i_f)] - i_f_true[~np.isnan(i_f_true)])),
                       0, atol=10 ** -5)


def test_nan_in_x():
    """
    Assure that time-resolved timefrequency functions do not return all NaN
    if one of the elements in the input array is NaN.
    Do this by replacing edge artifacts with NaN for a lowpass filter
    """

    # Generate a low-pass filtered signal with NaNs
    sig = np.random.randn(10000)
    fs = 1000
    sig = filter_signal(sig, fs, 'lowpass', fc=50)

    # Compute phase, amp, and freq time series
    f_range = (4, 8)
    pha = phase_by_time(sig, fs, f_range)
    amp = amp_by_time(sig, fs, f_range)
    i_f = freq_by_time(sig, fs, f_range)

    assert len(pha[~np.isnan(pha)]) > 0
    assert len(amp[~np.isnan(amp)]) > 0
    assert len(i_f[~np.isnan(i_f)]) > 0
