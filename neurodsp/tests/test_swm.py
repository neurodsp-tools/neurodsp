"""
test_swm.py
Test the sliding window matching function
"""

import pytest
import numpy as np
import os
import neurodsp
from neurodsp import shape
from neurodsp.tests import _load_example_data


def test_swm_consistent():
    """
    Confirm consistency in beta bandpass filter results on a neural signal
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    Fs = 1000

    # Load ground truth lagged coherence
    avg_window_true = np.load(os.path.dirname(neurodsp.__file__) + '/tests/data/sample_data_'+str(data_idx)+'_swm.npy')

    # Compute lagged coherence
    L = .055
    G = .2
    np.random.seed(1)
    avg_window, _, _ = shape.sliding_window_matching(x, Fs, L, G, max_iterations=500)

    # Compute difference between current and past signals
    signal_diff = avg_window - avg_window_true
    assert np.allclose(np.sum(np.abs(signal_diff)), 0, atol=10 ** -5)
