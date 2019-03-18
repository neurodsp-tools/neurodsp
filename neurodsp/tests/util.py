"""Utility functions for testing neurodsp functions."""

import os
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt

import neurodsp

###################################################################################################
###################################################################################################

def generate_random_signal(len_sig=2000, seed=0):
    """Generate a random time series"""

    np.random.seed(seed)
    sig = np.random.randn(len_sig)

    return sig


def load_example_data(data_idx=1, filtered=False):
    """Load an example voltage time series collected experimentally"""

    # Load time series
    sig = np.load(os.path.dirname(neurodsp.__file__) + \
                  '/tests/data/sample_data_'+str(data_idx)+'.npy')

    # Load ground-truth filtered data
    if filtered:
        sig_filt_true = np.load(os.path.dirname(neurodsp.__file__) + \
            '/tests/data/sample_data_'+str(data_idx)+'_filt.npy')
        return sig, sig_filt_true
    else:
        return sig


def plot_test(func):
    """Decorator for simple testing of plotting functions.

    Notes
    -----
    This decorator closes all plots prior to the test.
    After running the test function, it checks an axis was created with data.
    It therefore performs a minimal test - asserting the plots exists, with no accuracy checking.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        plt.close('all')

        func(*args, **kwargs)

        ax = plt.gca()
        assert ax.has_data()

    return wrapper
