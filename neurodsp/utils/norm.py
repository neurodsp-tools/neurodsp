"""Normalization related utility functions."""

import numpy as np

###################################################################################################
###################################################################################################

def normalize_sig(sig, mean=None, variance=None):
    """Normalize the mean and variance of a signal.

    Parameters
    ----------
    sig : 1d array
        Signal to normalize.
    variance : float, optional
        Variance to normalize to.
    mean : float, optional
        New mean for data to have.

    Returns
    -------
    sig : 1d array
        Input signal, with normalizations applied.
    """

    # Apply variance & mean transformations
    if variance is not None:
        sig = normalize_variance(sig, variance=variance)
    if mean is not None:
        sig = demean(sig, mean=mean)

    return sig


def demean(array, mean=0.):
    """Demean an array, updating to specified mean.

    Parameters
    ----------
    array : 1d array
        Data to demean.
    mean : float, optional, default: 0
        New mean for data to have.

    Returns
    -------
    out : 1d array
        Demeaned data.
    """

    return array - array.mean() + mean


def normalize_variance(array, variance=1.):
    """Normalize the variance of an array, updating to specified variance.

    Parameters
    ----------
    array : 1d array
        Data to normalize variance to.
    variance : float, optional, default: 1.0
        Variance to normalize to.

    Returns
    -------
    out : 1d array
        Variance normalized data.

    Notes
    -----
    If the input array is all zeros, this function simply returns the input.
    """

    # If array is all zero, set to return the same array (can't update variance of 0)
    if not array.any():
        out = array

    else:
        out = array / array.std() * np.sqrt(variance)

    return out
