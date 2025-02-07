"""Autocorrelation related analyses of time series."""

import numpy as np
from scipy.optimize import curve_fit

from neurodsp.utils.decorators import multidim

###################################################################################################
###################################################################################################

@multidim()
def compute_autocorr(sig, max_lag=1000, lag_step=1, demean=True):
    """Compute the signal autocorrelation (lagged correlation).

    Parameters
    ----------
    sig : array
        Time series to compute autocorrelation over.
    max_lag : int, optional, default: 1000
        Maximum delay to compute autocorrelations for, in samples.
    lag_step : int, optional, default: 1
        Step size (lag advance) for computing autocorrelations.
    demean : bool, optional, default: True
        Whether to demean the signal before computing autocorrelations.

    Returns
    -------
    timepoints : 1d array
        Time points, in samples, at which autocorrelations are computed.
    autocorrs : array
        Autocorrelation values, across time lags.

    Examples
    --------
    Compute the autocorrelation of a simulated pink noise signal:

    >>> from neurodsp.sim import sim_powerlaw
    >>> sig = sim_powerlaw(n_seconds=10, fs=500, exponent=-1)
    >>> timepoints, autocorrs = compute_autocorr(sig)
    """

    if demean:
        sig = sig - sig.mean()

    autocorrs = np.correlate(sig, sig, "full")[len(sig)-1:]
    autocorrs = autocorrs[:max_lag+1] / autocorrs[0]
    autocorrs = autocorrs[::lag_step]

    timepoints = np.arange(0, max_lag+1, lag_step)

    return timepoints, autocorrs


def compute_decay_time(timepoints, autocorrs, fs, level=0):
    """Compute autocorrelation decay time, from precomputed autocorrelation.

    Parameters
    ----------
    timepoints : 1d array
        Timepoints for the computed autocorrelations.
    autocorrs : 1d array
        Autocorrelation values.
    fs : int
        Sampling rate of the signal.
    level : float
        Autocorrelation decay threshold.

    Returns
    -------
    result : float
        Autocorrelation decay time.
        If decay time value not found, returns nan.

    Notes
    -----
    The autocorrelation decay time is computed as the time delay for the
    autocorrelation to drop to (or below) the decay time threshold.
    """

    val_checks = autocorrs <= level

    if np.any(val_checks):
        # Get the first value to cross the threshold, and convert to time value
        result = timepoints[np.argmax(val_checks)] / fs
    else:
        result = np.nan

    return result


def fit_autocorr(timepoints, autocorrs, fit_function='single_exp', bounds=None):
    """Fit autocorrelation function, returning timescale estimate.

    Parameters
    ----------
    timepoints : 1d array
        Timepoints for the computed autocorrelations, in samples or seconds.
    autocorrs : 1d array
        Autocorrelation values.
    fs : int, optional
        Sampling rate of the signal.
        If provided, timepoints are converted to time values.
    fit_func : {'single_exp', 'double_exp'}
        Which fitting function to use to fit the autocorrelation results.
    bounds : tuple of list
        Parameter bounds for fitting.
        Organized as ([min_p1, min_p1, ...], [max_p1, max_p2, ...]).

    Returns
    -------
    popts
        Fit parameters. Parameters depend on the fitting function.
        If `fit_func` is 'single_exp', fit parameters are: tau, scale, offset
        If `fit_func` is 'douple_exp', fit parameters are: tau1, tau2, scale1, scale2, offset
        See fit function for more details.

    Notes
    -----
    The values / units of the returned parameters are dependent on the units of samples.
    For example, if the timepoints input is in samples, the fit tau value is too.
    If providing parameter bounds, these also need to match the unit of timepoints.
    """

    if not bounds:
        if fit_function == 'single_exp':
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        elif fit_function == 'double_exp':
            bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

    popts, _ = curve_fit(AC_FIT_FUNCS[fit_function], timepoints, autocorrs, bounds=bounds)

    return popts


## AC FITTING

def exp_decay_func(timepoints, tau, scale, offset):
    """Exponential decay fit function.

    Parameters
    ----------
    timepoints : 1d array
        Time values.
    tau : float
        Timescale value.
    scale : float
        Scaling factor, which captures the start value of the function.
    offset : float
        Offset factor, which captures the end value of the function.

    Returns
    -------
    ac_fit : 1d array
        Result of fitting the function to the autocorrelation.
    """

    return scale * (np.exp(-timepoints / tau) + offset)


def double_exp_decay_func(timepoints, tau1, tau2, scale1, scale2, offset):
    """Exponential decay fit function with two timescales.

    Parameters
    ----------
    timepoints : 1d array
        Time values.
    tau1, tau2 : float
        Timescale values, for the 1st and 2nd timescale.
    scale1, scale2 : float
        Scaling factors, for the 1st and 2nd timescale.
    offset : float
        Offset factor.

    Returns
    -------
    ac_fit : 1d array
        Result of fitting the function to the autocorrelation.
    """

    return scale1 * np.exp(-timepoints / tau1) + scale2 * np.exp(-timepoints / tau2) + offset


AC_FIT_FUNCS = {
    'single_exp' : exp_decay_func,
    'double_exp' : double_exp_decay_func,
}


def compute_ac_fit(timepoints, *popts, fit_function='single_exp'):
    """Regenerate values of the exponential decay fit.

    Parameters
    ----------
    timepoints : 1d array
        Time values, in samples or seconds.
    *popts
        Fit parameters.
    fit_func : {'single_exp', 'double_exp'}
        Which fit function to use to fit the autocorrelation results.

    Returns
    -------
    fit_values : 1d array
        Values of the fit to the autocorrelation values.
    """

    fit_func = AC_FIT_FUNCS[fit_function]

    return fit_func(timepoints, *popts)
