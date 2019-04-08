"""Utility functions for neurodsp."""

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


def remove_nans(sig):
    """Drop any NaNs on the edges of a 1d array.

    Parameters
    ----------
    sig : 1d array
        Signal to be checked for edge NaNs.

    Returns
    -------
    sig_removed : 1d array
        Signal with NaN edges removed.
    sig_nans : 1d array
        Boolean array indicating where NaNs were in the original array.
    """

    sig_nans = np.isnan(sig)
    sig_removed = sig[np.where(~np.isnan(sig))]

    return sig_removed, sig_nans


def restore_nans(sig, sig_nans, dtype=float):
    """Restore NaN values to the edges of a 1d array.

    Parameters
    ----------
    sig : 1d array
        Signal that has had NaN edges removed.
    sig_nans : 1d array
        Boolean array indicating where NaNs were in the original array.

    Returns
    -------
    sig_restored : 1d array
        Signal with NaN edges restored.
    """

    sig_restored = np.ones(len(sig_nans), dtype=dtype) * np.nan
    sig_restored[~sig_nans] = sig

    return sig_restored


def discard_outliers(data, outlier_percent):
    """Discard outlier arrays with high values.

    Parameters
    ----------
    data : nd array
        Array to remove outliers from.
    outlier_percent : float
        The percentage of outlier values to be removed.

    Returns
    -------
    data : nd array
        Array after removing outliers.
    """

    # Get the number of arrays to discard - round up so it doesn't get a zero.
    n_discard = int(np.ceil(data.shape[-1] / 100. * outlier_percent))

    # Make 2D -> 3D for looping across array
    data = data[np.newaxis, :, :] if data.ndim == 2 else data

    # Select the windows to keep from each 2D component of the input data
    data = [dat[:, np.argsort(np.mean(np.log10(dat), axis=0))[:-n_discard]] for dat in data]

    # Reshape array and squeeze to drop back to 2D if that was original shape
    data = np.squeeze(np.stack(data))

    return data
