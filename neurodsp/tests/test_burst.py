"""
test_burst.py
Test burst detection functions
"""

import os
import pytest
import numpy as np
import neurodsp
from neurodsp.tests import _load_example_data


def test_detect_bursts_consistent():
    """
    Confirm consistency in burst detection results on a generated neural signal
    """
    # Load data and ground-truth filtered signal
    x = _load_example_data(data_idx=1)
    Fs = 1000
    f_range = (13, 30)
    f_oi = np.floor(np.mean(f_range))
    f_range_slope = (3, 50)
    f_slope_excl = f_range

    # Load past burst findings
    bursting_true_deviation = np.load(os.path.dirname(neurodsp.__file__) +
                                      '/tests/data/sample_data_1_burst_deviation.npy')
    bursting_true_bosc = np.load(os.path.dirname(neurodsp.__file__) +
                                 '/tests/data/sample_data_1_burst_bosc.npy')

    # Detect bursts with different algorithms
    bursting_deviation = neurodsp.detect_bursts(x, Fs, f_range, 'deviation',
                                                dual_thresh=(0.9, 2.0))
    bursting_bosc = neurodsp.detect_bursts_bosc(x, Fs, f_oi, f_range_slope, f_slope_excl)

    assert np.isclose(np.sum(bursting_deviation - bursting_true_deviation), 0)
    assert np.isclose(np.sum(bursting_bosc - bursting_true_bosc), 0)
