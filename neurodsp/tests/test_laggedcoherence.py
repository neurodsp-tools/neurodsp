"""Test function to compute lagged coherence."""

import os

import numpy as np

import neurodsp
from neurodsp.laggedcoherence import lagged_coherence
from .util import load_example_data

###################################################################################################
###################################################################################################

def test_laggedcoherence_consistent():

    # Load data
    data_idx = 1
    sig = load_example_data(data_idx=data_idx)
    fs = 1000
    f_range = (13, 30)

    # Load ground truth lagged coherence
    lc_true = np.load(os.path.dirname(neurodsp.__file__) +
                      '/tests/data/sample_data_' + str(data_idx) + '_laggedcoherence.npy')

    # Compute lagged coherence
    lag_coh_beta = lagged_coherence(sig, f_range, fs)

    assert np.allclose(lag_coh_beta, lc_true, atol=10 ** -5)
