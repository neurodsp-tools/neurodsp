"""
test_cyclepoints.py
Test identification of extrema and zerocrossings
"""

import numpy as np
import os
import neurodsp
from neurodsp import shape
from neurodsp.tests import _load_example_data


def test_Ps_consistent():
    """
    Confirm consistency in peak finding
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    Fs = 1000
    f_range = (13, 30)

    # Load ground truth lagged coherence
    Ps_true = np.load(os.path.dirname(neurodsp.__file__) +
                      '/tests/data/sample_data_' + str(data_idx) + '_Ps.npy')

    # Compute lagged coherence
    Ps, Ts = shape.find_extrema(x, Fs, f_range)

    # Compute difference between current and past signals
    signal_diff = Ps - Ps_true
    assert np.allclose(np.sum(np.abs(signal_diff)), 0, atol=10 ** -5)


def test_zeroxR_consistent():
    """
    Confirm consistency in peak finding
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    Fs = 1000
    f_range = (13, 30)

    # Load ground truth lagged coherence
    zeroxR_true = np.load(os.path.dirname(neurodsp.__file__) +
                          '/tests/data/sample_data_' + str(data_idx) + '_zeroxR.npy')

    # Compute lagged coherence
    Ps, Ts = shape.find_extrema(x, Fs, f_range)
    zeroxR, zeroxD = shape.find_zerox(x, Ps, Ts)

    # Compute difference between current and past signals
    signal_diff = zeroxR - zeroxR_true
    assert np.allclose(np.sum(np.abs(signal_diff)), 0, atol=10 ** -5)

