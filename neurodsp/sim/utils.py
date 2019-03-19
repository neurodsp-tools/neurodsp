"""Utility functions for simulations."""

import numpy as np

###################################################################################################
###################################################################################################

def normalized_sum(sig1, sig2, ratio):
    """Combine two signals, after transforming to have the desired variance ratio.

    Parameters
    ----------
    sig1, sig2 : 1d array
        Vectors of data to normalize variance with respect to each other.
    ratio : float
        Desired ratio of sig1 variance to sig2 variance.
        If > 1 - sig1 is stronger, if < 1 - sig2 is stronger.

    Returns
    -------
    1d array
        New vector with the combined input signals.
    """

    return sum(normalize_variance(sig1, sig2, ratio))


def normalize_variance(sig1, sig2, ratio):
    """Normalize the variance across two signals to reflect a specified ratio.

    Parameters
    ----------
    sig1, sig2 : 1d array
        Vectors of data to normalize variance with respect to each other.
    ratio : float
        Desired ratio of sig1 variance to sig2 variance.
        If > 1 - sig1 is stronger, if < 1 - sig2 is stronger.

    Returns
    -------
    1d array, 1d array
        New sig1 & sig2, where sig2 has been normalized so they have the desired variance ratio.
    """

    return sig1, np.sqrt(sig2**2 * np.var(sig1) / (np.var(sig2) * ratio)) * np.sign(sig2)
