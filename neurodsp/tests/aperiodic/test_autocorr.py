"""Tests for neurodsp.aperiodic.autocorr."""

from neurodsp.tests.settings import FS

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
