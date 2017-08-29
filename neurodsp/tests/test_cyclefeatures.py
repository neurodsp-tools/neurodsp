"""
test_cyclefeatures.py
Test measurement of cycle-by-cycle features of oscillatory waveforms
"""

import numpy as np
import pandas as pd
import os
import neurodsp
from neurodsp import shape
from neurodsp.tests import _load_example_data


def test_cyclefeatures_consistent():
    """
    Confirm consistency in peak finding
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    Fs = 1000
    f_range = (13, 30)

    # Load ground truth lagged coherence
    df_true = pd.read_csv(os.path.dirname(neurodsp.__file__) +
                          '/tests/data/sample_data_' + str(data_idx) + '_cyclefeatures.csv')

    # Compute lagged coherence
    true_oscillating_periods_kwargs = {'restrict_by_amplitude_consistency': False,
                                       'restrict_by_period_consistency': False,
                                       'amplitude_fraction_threshold': .3}

    df = shape.features_by_cycle(x, Fs, f_range, center_extrema='T',
                                 estimate_oscillating_periods=True,
                                 true_oscillating_periods_kwargs=true_oscillating_periods_kwargs)

    # Compute difference between calculated and ground truth values for each column
    for k in df.keys():
        signal_diff = df[k].values - df_true[k].values
        assert np.allclose(np.sum(np.abs(signal_diff)), 0, atol=10 ** -5)
