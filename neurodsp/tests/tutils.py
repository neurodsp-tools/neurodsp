"""Utility functions for testing neurodsp functions."""

from functools import wraps

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.utils.data import compute_nsamples

from neurodsp.tests.settings import N_SECONDS, FS

###################################################################################################
###################################################################################################

def check_sim_output(sig, n_seconds=None, fs=None):
    """Helper function to check some basic properties of simulated signals."""

    n_seconds = N_SECONDS if not n_seconds else n_seconds
    fs = FS if not fs else fs
    exp_n_samples = compute_nsamples(n_seconds, fs)

    assert isinstance(sig, np.ndarray)
    assert len(sig) == exp_n_samples
    assert sum(np.isnan(sig)) == 0

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

def check_exponent(xs, offset, exp):
    return xs*exp + offset
