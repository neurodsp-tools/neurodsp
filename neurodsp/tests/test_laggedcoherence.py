"""
test_laggedcoherence.py
Test function to compute lagged coherence
"""

import numpy as np
import os
import neurodsp
from neurodsp.tests import _load_example_data


def test_laggedcoherence_consistent():
    """
    Confirm consistency in beta bandpass filter results on a neural signal
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    Fs = 1000
    f_range = (13, 30)

    # Load ground truth lagged coherence
    lc_true = np.load(os.path.dirname(neurodsp.__file__) +
                      '/tests/data/sample_data_' + str(data_idx) + '_laggedcoherence.npy')

    # Compute lagged coherence
    lag_coh_beta = neurodsp.lagged_coherence(x, f_range, Fs)

    assert np.allclose(lag_coh_beta, lc_true, atol=10 ** -5)
