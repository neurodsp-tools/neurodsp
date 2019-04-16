"""Utility functions for testing neurodsp functions."""

from functools import wraps

import numpy as np
import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

def get_random_signal(len_sig=1000, seed=0):
    """Generate a random time series for testing."""

    np.random.seed(seed)
    sig = np.random.randn(len_sig)

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
