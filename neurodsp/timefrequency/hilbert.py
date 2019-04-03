"""Hilbert transforms."""

import numpy as np
from scipy.signal import hilbert

from neurodsp.utils import remove_nans, restore_nans

###################################################################################################
###################################################################################################

def robust_hilbert(sig, increase_n=False):
    """Compute the hilbert transform, ignoring the boundaries of that are filled with NaN.

    Parameters
    ----------
    sig : xx
        xx
    increase_n : x
        xx

    Returns
    -------
    sig_hilb : x
        xx
    """

    # Extract the signal that is not nan
    sig_nonan, sig_nans = remove_nans(sig)

    # Compute hilbert transform of signal without nans
    if increase_n:
        sig_len = len(sig_nonan)
        n_components = 2**(int(np.log2(sig_len)) + 1)
        sig_hilb_nonan = hilbert(sig_nonan, n_components)[:sig_len]
    else:
        sig_hilb_nonan = hilbert(sig_nonan)

    # Fill in output hilbert with nans on edges
    sig_hilb = restore_nans(sig_hilb_nonan, sig_nans, dtype=complex)

    return sig_hilb
