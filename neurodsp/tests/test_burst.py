"""
test_burst.py
Test burst detection functions

Code to make true data for compute_burst_stats:

import os
import numpy as np
import neurodsp
from neurodsp.burst import compute_burst_stats
import pickle

fs = 1000
bursting_true = np.load(os.path.dirname(neurodsp.__file__) +
                            '/tests/data/sample_data_1_burst_deviation.npy')
burst_stats = compute_burst_stats(bursting_true, fs)

with open((os.path.dirname(neurodsp.__file__) + '/tests/data/sample_data_1_burst_stats.pkl'), 'wb') as f:
    pickle.dump(burst_stats, f, pickle.HIGHEST_PROTOCOL)
"""

import os
import pickle

import numpy as np

import neurodsp
from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats
from .util import load_example_data

###################################################################################################
###################################################################################################

def test_detect_bursts_dual_threshold():
    """Confirm consistency in burst detection results on a generated neural signal."""

    # Load data and ground-truth filtered signal
    sig = load_example_data(data_idx=1)
    fs = 1000
    f_range = (13, 30)

    # Load past burst findings
    bursting_true = np.load(os.path.dirname(neurodsp.__file__) +
                            '/tests/data/sample_data_1_burst_deviation.npy')

    # Detect bursts with different algorithms
    bursting = detect_bursts_dual_threshold(sig, fs, f_range, (0.9, 2))

    assert np.isclose(np.sum(bursting - bursting_true), 0)


def test_compute_burst_stats():
    """Confirm consistency in burst detection results on a generated neural signal."""

    # compute burst stats
    fs = 1000
    bursting_true = np.load(os.path.dirname(neurodsp.__file__) +
                            '/tests/data/sample_data_1_burst_deviation.npy')
    burst_stats = compute_burst_stats(bursting_true, fs)

    # Load true burst stats
    with open((os.path.dirname(neurodsp.__file__) + '/tests/data/sample_data_1_burst_stats.pkl'), 'rb') as f:
        burst_stats_true = pickle.load(f)

    # Compare new and old burst stats
    for k in burst_stats_true.keys():
        assert np.isclose(burst_stats_true[k], burst_stats[k])
