"""Tests for neurodsp.aperiodic.autocorr."""

from neurodsp.tests.tsettings import FS

from neurodsp.aperiodic.autocorr import *

###################################################################################################
###################################################################################################

def test_compute_autocorr(tsig):

    max_lag = 500
    timepoints, autocorrs = compute_autocorr(tsig, max_lag=max_lag)
    assert len(timepoints) == len(autocorrs) == max_lag + 1

def test_compute_decay_time(tsig):

    timepoints, autocorrs = compute_autocorr(tsig, max_lag=500)
    decay_time = compute_decay_time(timepoints, autocorrs, FS)
    assert isinstance(decay_time, float)

def test_fit_autocorr(tsig):
    # This is a smoke test - check it runs with no accuracy checking

    timepoints, autocorrs = compute_autocorr(tsig, max_lag=500)

    popts1 = fit_autocorr(timepoints, autocorrs, fit_function='single_exp')
    fit_vals1 = compute_ac_fit(timepoints, *popts1, fit_function='single_exp')
    assert np.all(fit_vals1)

    # Test with bounds passed in
    bounds = ([0, 0, 0], [10, np.inf, np.inf])
    popts1 = fit_autocorr(timepoints, autocorrs, 'single_exp', bounds)
    fit_vals1 = compute_ac_fit(timepoints, *popts1, fit_function='single_exp')
    assert np.all(fit_vals1)

    popts2 = fit_autocorr(timepoints, autocorrs, fit_function='double_exp')
    fit_vals2 = compute_ac_fit(timepoints, *popts2, fit_function='double_exp')
    assert np.all(fit_vals2)

def test_fit_autocorr_acc():
    # This test includes some basic accuracy checking

    fs = 100
    tau = 0.015
    lags = np.linspace(0, 100, 1000)

    # Test can fit and recompute the tau value
    corrs1 = exp_decay_func(lags, tau, 1, 0)
    params1 = fit_autocorr(lags, corrs1)
    assert np.isclose(params1[0], tau, 0.001)

    # Test can fit and recompute the tau value - timepoints as time values
    lags = lags / fs
    corrs2 = exp_decay_func(lags, tau, 1, 0)
    params2 = fit_autocorr(lags, corrs2)
    assert np.isclose(params2[0], tau, 0.001)

def test_compute_autocorr_short_signal():
    """Lag coordinates match direct correlations even beyond the available data."""
    sig = np.array([1., 3., 2., -1., 4.])
    centered = sig - sig.mean()
    energy = np.dot(centered, centered)
    for max_lag in [4, 5, 1000]:
        for lag_step in [1, 2, 7]:
            times, values = compute_autocorr(sig, max_lag=max_lag, lag_step=lag_step)
            expected_times = np.arange(0, len(sig), lag_step)
            expected_values = np.array([
                np.dot(centered[:len(sig)-lag], centered[lag:]) / energy
                for lag in expected_times
            ])
            np.testing.assert_array_equal(times, expected_times)
            np.testing.assert_allclose(values, expected_values)


def test_compute_autocorr_short_fit():
    """Default lags can feed the documented exponential fit on a short recording."""
    rng = np.random.default_rng(0)
    sig = np.convolve(rng.standard_normal(300), np.exp(-np.arange(20) / 3.))[:300]
    times, values = compute_autocorr(sig)
    params = fit_autocorr(times, values)
    assert np.isfinite(params).all()
