"""Normalization related utility functions for neurodsp."""

import numpy as np

###################################################################################################
###################################################################################################

def demean(array, mean=0., select_nonzero=True):
    """Demean an array, updating to specified mean.

    Parameters
    ----------
    array : 1d array
        Data to demean.
    mean : float, optional, default: 0
        New mean for data to have.
    select_nonzero : boolean, optional, default=True
        Whether to calculate the mean of the array across only non-zero data points.

    Returns
    -------
    out : 1d array
        Demeaned data.
    """

    if select_nonzero:
        nonzero = np.nonzero(array)
        nonzero_mean = np.mean(array[nonzero])
        out = array.copy()
        out[nonzero] = array[nonzero] - nonzero_mean + mean

    else:
        out = array - array.mean() + mean

    return out


def normalize_variance(array, variance=1., select_nonzero=True):
    """Normalize the variance of an array, updating to specified variance.

    Parameters
    ----------
    array : 1d array
        Data to normalize variance to.
    variance : float, optional, default=1.
        Variance to normalize to.
    select_nonzero : boolean, optional, default=True
        Whether to calculate the variance of the array across only non-zero data points.

    Returns
    -------
    out : 1d array
        Variance normalized data.

    Notes
    -----
    If the input array is all zeros, this function simply returns the input.
    """

    # If array is all zero, set to return the same array
    #   Can't update variance if it's zero
    if not array.any():
        out = array

    else:

        if select_nonzero:
            nonzero = np.nonzero(array)
            array_std = np.std(array[nonzero])
            out = array.copy()
            out[nonzero] = array[nonzero] / array_std * np.sqrt(variance)
        else:
            out = array / array.std() * np.sqrt(variance)

    return out
