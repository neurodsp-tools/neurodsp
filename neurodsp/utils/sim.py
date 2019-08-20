"""Simulation related utility functions."""

import numpy as np

###################################################################################################
###################################################################################################

def set_random_seed(seed_val=0):
    """Set the random seed value.

    Parameters
    ----------
    seed_val : int
        Value to set the random seed as.
    """

    np.random.seed(seed_val)
