"""Outlier & missing data related utility functions for NeuroDSP."""

import numpy as np

###################################################################################################
###################################################################################################

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
