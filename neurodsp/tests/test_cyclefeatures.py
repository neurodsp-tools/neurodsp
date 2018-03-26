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

    Previous code run:
    import numpy as np
    import os
    import neurodsp
    from neurodsp import shape
    x = np.load(os.path.dirname(neurodsp.__file__) + '/tests/data/sample_data_1.npy')
    df = shape.features_by_cycle(x, 1000, (13, 30), center_extrema='T',
                                 estimate_oscillating_periods=True)
    df.to_csv(os.path.dirname(neurodsp.__file__) + '/tests/data/sample_data_1_cyclefeatures.csv')
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    Fs = 1000
    f_range = (13, 30)

    # Load ground truth cycle features
    df_true = pd.read_csv(os.path.dirname(neurodsp.__file__) +
                          '/tests/data/sample_data_' + str(data_idx) + '_cyclefeatures.csv')

    df = shape.features_by_cycle(x, Fs, f_range, center_extrema='T',
                                 estimate_oscillating_periods=True)

    # Compute difference between calculated and ground truth values for each column
    for k in df.keys():
        signal_diff = df[k].values[~np.isnan(df[k].values)] - df_true[k].values[~np.isnan(df_true[k].values)]
        assert np.allclose(np.sum(np.abs(signal_diff)), 0, atol=10 ** -5)
