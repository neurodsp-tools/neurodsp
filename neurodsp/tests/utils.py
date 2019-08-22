"""Utility functions for testing neurodsp functions."""

from functools import wraps

import numpy as np
import matplotlib.pyplot as plt

###################################################################################################
###################################################################################################

# Some default settings for the sim tests
FS = 100
N_SECONDS = 0.75
FREQ = 10

###################################################################################################
###################################################################################################

def check_sim_output(sig):
    """Helper function to check some basic properties of simualated signals."""

    exp_n_samples = int(FS * N_SECONDS)

    assert isinstance(sig, np.ndarray)
    assert len(sig) == exp_n_samples
    assert sum(np.isnan(sig)) == 0

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
