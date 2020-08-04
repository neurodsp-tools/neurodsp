"""Autocorrelation analyses of time series."""

import numpy as np

###################################################################################################
###################################################################################################

def compute_autocorr(sig, max_lag=1000, lag_step=1, demean=True):
    """Compute the signal autocorrelation (lagged correlation).

    Parameters
    ----------
    sig : array 1D
        Time series to compute autocorrelation over.
    max_lag : int, optional, default: 1000
        Maximum delay to compute autocorrelations for, in samples.
    lag_step : int, optional, default: 1
        Step size (lag advance) for computing autocorrelations.
    demean : bool, optional, default: True
        Whether to demean the signal before computing autcorralations.

    Returns
    -------
    timepoints : 1d array
        Time points, in samples, at which autocorrelations are computed.
    autocorrs : 1d array
        Autocorrelation values, for across time lags.
    """

    if demean:
        sig = sig - sig.mean()

    autocorrs = np.correlate(sig, sig, "full")[len(sig)-1:]
    autocorrs = autocorrs[:max_lag] / autocorrs[0]
    autocorrs = autocorrs[::lag_step]

    timepoints = np.arange(0, max_lag, lag_step)

    return timepoints, autocorrs
