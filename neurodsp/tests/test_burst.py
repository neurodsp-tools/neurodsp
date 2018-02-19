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
    bursting_true = np.load(os.path.dirname(neurodsp.__file__) +
                            '/tests/data/sample_data_1_burst_deviation.npy')

    # Detect bursts
    Fs = 1000
    f_lo = 13
    f_hi = 30

    bursting = neurodsp.detect_bursts(x, Fs, (f_lo, f_hi,),
                                      algorithm='deviation', thresh=(0.9, 2.0))
    assert np.isclose(np.sum(bursting - bursting_true), 0)
