"""Outlier & missing data related utility functions."""

import numpy as np

###################################################################################################
###################################################################################################

def remove_nans(sig):
    """Drop any NaNs on the edges of an array.

    Parameters
    ----------
    sig : 1d or 2d array
        Signal to be checked for edge NaNs.

    Returns
    -------
    sig_removed : 1d or 2d array
        Signal with NaN edges removed.
    sig_nans : 1d array
        Boolean array indicating where NaNs were in the original array.

    Notes
    -----
    For 2d arrays, this function assumes the same columns to be NaN across all rows.
    """

    sig_nans = np.isnan(sig)

    if sig.ndim == 1:
        sig_removed = sig[np.where(~sig_nans)]
    elif sig.ndim == 2:
        sig_removed = sig[~sig_nans].reshape(sig_nans.shape[0], sum(~sig_nans[0, :]))
        sig_nans = sig_nans[0, :]
    else:
        raise ValueError('Only 1d or 2d arrays supported.')

    return sig_removed, sig_nans


def restore_nans(sig, sig_nans, dtype=float):
    """Restore NaN values to the edges of an array.

    Parameters
    ----------
    sig : 1d or 2d array
        Signal that has had NaN edges removed.
    sig_nans : 1d array
        Boolean array indicating where NaNs were in the original array.

    Returns
    -------
    sig_restored : 1d or 2d array
        Signal with NaN edges restored.

    Notes
    -----
    If sig is 2d, the sig_nans input should reflect the values for a row.
    This function assumes the same columns to be NaN across all rows.
    """

    if sig.ndim == 1:
        sig_restored = np.ones(len(sig_nans), dtype=dtype) * np.nan
        sig_restored[~sig_nans] = sig
    elif sig.ndim == 2:
        sig_restored = np.ones([sig.shape[0], len(sig_nans)], dtype=dtype) * np.nan
        sig_restored[:, np.where(sig_nans == False)[0]] = sig
    else:
        raise ValueError('Only 1d or 2d arrays supported.')

    return sig_restored


def discard_outliers(data, outlier_percent):
    """Discard outlier arrays with high values.

    Parameters
    ----------
    data : 2d or 3d array
        Array to remove outliers from.
    outlier_percent : float
        The percentage of outlier values to be removed. Must be between 0 and 100.

    Returns
    -------
    data : array
        Array after removing outliers.

    Notes
    -----
    This function drops entries across the last dimension.
    Values are dropped based on being an outlier in log10 spacing.
    """

    # Get the number of arrays to discard - round up so it doesn't get a zero.
    n_discard = int(np.ceil(data.shape[-1] / 100. * outlier_percent))

    # Check discard settings compared to data size
    if n_discard >= data.shape[-1]:
        raise ValueError('Outlier removal would discard all data. Can not proceed.')

    # Make 2D -> 3D for looping across array
    data = data[np.newaxis, :, :] if data.ndim == 2 else data

    # Select the windows to keep from each 2D component of the input data
    data = [dat[:, np.argsort(np.mean(np.log10(dat), axis=0))[:-n_discard]] for dat in data]

    # Reshape array and squeeze to drop back to 2D if that was original shape
    data = np.squeeze(np.stack(data))

    # Ensure output maintains the correct shape, keeping 2D if ends up as 1D
    if data.ndim == 1:
        data = data[:, np.newaxis]

    return data
