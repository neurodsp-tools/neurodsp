"""Utility functions for simulations."""

import numpy as np

###################################################################################################
###################################################################################################

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
    1d array
        Demeaned data.
    """

    return array - array.mean() + mean


def normalize_variance(array, variance=1.):
    """Normalize the variance of an array, updating to specified variance.

    Parameters
    ----------
    array : 1d array
        Data to normalize variance to.
    variance : float, optional, default=1.
        Variance to normalize to.

    Returns
    -------
    1d array
        Variance normalized data.
    """

    return array / array.std() * np.sqrt(variance)


def proportional_sum(signals, proportions):
    """Sum a set of signals, each with a specified proportional variance.

    Parameters
    ----------
    signals : list of 1d array
        Signals to sum together.
    proportions : list of float
        Proportional variance for each signal.

    Returns
    -------
    1d array
        Sumated signal.
    """

    return sum([normalize_variance(sig, prop) for prop, sig in zip(proportions, signals)])


# def normalized_sum(sig1, sig2, ratio, select_nonzero=True):
#     """Combine two signals, after transforming to have the desired variance ratio.

#     Parameters
#     ----------
#     sig1, sig2 : 1d array
#         Arrays of data to normalize variance with respect to each other.
#     ratio : float
#         Desired ratio of sig1 variance to sig2 variance.
#         If > 1 - sig1 is stronger, if < 1 - sig2 is stronger.
#     select_nonzero : boolean, optional, default=True
#         Whether to calculate the variance of sig1 across only non-zero data points.

#     Returns
#     -------
#     1d array
#         New array with the combined input signals.
#     """

#     return sum(variance_ratio(sig1, sig2, ratio, select_nonzero))


# def variance_ratio(sig1, sig2, ratio, select_nonzero=True):
#     """Normalize the variance across two signals to reflect a specified ratio.

#     Parameters
#     ----------
#     sig1, sig2 : 1d array
#         Vectors of data to normalize variance with respect to each other.
#     ratio : float
#         Desired ratio of sig1 variance to sig2 variance.
#         If > 1 - sig1 is stronger, if < 1 - sig2 is stronger.
#     select_nonzero : boolean, optional, default=True
#         Whether to calculate the variance of sig1 across only non-zero data points.

#     Returns
#     -------
#     1d array, 1d array
#         New sig1 & sig2, where sig2 has been normalized so they have the desired variance ratio.

#     Notes
#     -----
#     Only sig2 is actually modified by this procedure, relative to the variance in sig1.
#     If sig1 is a potentially non-continuous periodic signal (for example, bursty), `select_nonzero`
#     allows for normalizing the variance across only segments of the signal with signal present.
#     """

#     if select_nonzero:
#         sig1_var = np.var(np.nonzero(sig1))
#     else:
#         sig1_var = np.var(sig1)

#     sig2 = np.sqrt(sig2**2 * sig1_var / (np.var(sig2) * ratio)) * np.sign(sig2)

#     return sig1, sig2
