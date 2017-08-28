"""
test_shapephase.py
Test the phase estimation technique based on extrema and zerocrossings
"""

import pytest
import numpy as np
import os
import neurodsp
from neurodsp import shape
from neurodsp.tests import _load_example_data


def test_interpolated_phase_consistent():
    """
    Confirm consistency in beta bandpass filter results on a neural signal
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    Fs = 1000
    f_range = (13, 30)

    # Load ground truth phase time series
    phaPTRD_true = np.load(os.path.dirname(neurodsp.__file__) + '/tests/data/sample_data_'+str(data_idx)+'_phaPTRD.npy')

    # Compute phase time series
    Ps, Ts = shape.find_extrema(x, Fs, f_range)
    zeroxR, zeroxD = shape.find_zerox(x, Ps, Ts)
    phaPTRD = shape.extrema_interpolated_phase(x, Ps, Ts, zeroxR=zeroxR, zeroxD=zeroxD)

    # Compute difference between current and past filtered signals
    signal_diff = phaPTRD - phaPTRD_true
    assert np.allclose(np.sum(np.abs(signal_diff)), 0, atol=10 ** -5)
