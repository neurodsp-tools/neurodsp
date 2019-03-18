"""Utility functions for neurodsp."""

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
