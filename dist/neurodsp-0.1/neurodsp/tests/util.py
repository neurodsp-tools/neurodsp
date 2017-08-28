"""
util.py
Utility functions for testing neurodsp functions.
"""

import neurodsp
import os
import numpy as np


def _generate_random_x(len_x=2000, seed=0):
    """Generate a random time series"""
    np.random.seed(seed)
    x = np.random.randn(len_x)
    return x


def _load_example_data(data_idx=1, filtered=False):
    """Load an example voltage time series collected experimentally"""

    # Load time series
    x = np.load(os.path.dirname(neurodsp.__file__) + '/tests/data/sample_data_'+str(data_idx)+'.npy')

    # Load ground-truth filtered data
    if filtered:
        x_filt_true = np.load(os.path.dirname(neurodsp.__file__) + '/tests/data/sample_data_'+str(data_idx)+'_filt.npy')
        return x, x_filt_true
    else:
        return x
