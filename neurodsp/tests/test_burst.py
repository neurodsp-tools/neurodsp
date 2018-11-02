"""
test_burst.py
Test burst detection functions
"""

import os
import numpy as np
import neurodsp
from .util import _load_example_data


def test_detect_bursts_dual_threshold():
    """
    Confirm consistency in burst detection results on a generated neural signal
    """
    # Load data and ground-truth filtered signal
    sig = _load_example_data(data_idx=1)
    fs = 1000
    f_range = (13, 30)

    # Load past burst findings
    bursting_true = np.load(os.path.dirname(neurodsp.__file__) +
                            '/tests/data/sample_data_1_burst_deviation.npy')

    # Detect bursts with different algorithms
    bursting = neurodsp.detect_bursts_dual_threshold(sig, fs, f_range, (0.9, 2))

    assert np.isclose(np.sum(bursting - bursting_true), 0)
