"""Autocorrelation analyses of time series."""

import numpy as np

###################################################################################################
###################################################################################################

def compute_autocorr(sig, max_lag=1000, lag_step=1):
    """Compute the signal autocorrelation (lagged correlation).

    Parameters
    ----------
    sig : array 1D
        Time series to compute autocorrelation over.
    max_lag : int (default=1000)
        Maximum delay to compute autocorrelations for, in samples.
    lag_step : int (default=1)
        Step size (lag advance) for computing correlations.

    Returns
    -------
    timepoints : 1d array
        Time points, in samples, at which autocorrelations are computed.
    autocorrs : 1d array
        Autocorrelation values, for across time lags.
    """

    timepoints = np.arange(0, max_lag, lag_step)
    autocorrs = np.zeros(len(timepoints))

    autocorrs[0] = np.sum((sig - np.mean(sig))**2)

    for ind, lag in enumerate(timepoints[1:]):
        autocorrs[ind + 1] = \
            np.sum((sig[:-lag] - np.mean(sig[:-lag])) * (sig[lag:] - np.mean(sig[lag:])))

    autocorrs = autocorrs / autocorrs[0]

    return timepoints, autocorrs
