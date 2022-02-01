"""Tests neurodsp.aperiodic.dfa."""

from pytest import raises

import numpy as np

from neurodsp.sim import sim_powerlaw

from neurodsp.tests.settings import N_SECONDS

from neurodsp.aperiodic.dfa import *

###################################################################################################
###################################################################################################

def test_compute_fluctuations(tsig):

    t_scales, flucs, result = compute_fluctuations(tsig, 500)
    assert len(t_scales) == len(flucs)

    # Check error if the settings create window lengths that are too short
    with raises(ValueError):
        t_scales, flucs, result = compute_fluctuations(tsig, 100)

    # Check error for nonsense method input
    with raises(ValueError):
        t_scales, flucs, result = compute_fluctuations(tsig, 500, method='nope.')

def test_compute_fluctuations_dfa():

    # Note: these tests need a high sampling rate, so we use local simulations

    # Test white noise: expected DFA of 0.5
    fs = 1000
    white = sim_powerlaw(N_SECONDS, fs, exponent=0)
    t_scales, flucs, alpha = compute_fluctuations(white, fs, method='dfa')
    assert np.isclose(alpha, 0.5, atol=0.1)

    # Test brown noise: expected DFA of 1.5
    brown = sim_powerlaw(N_SECONDS, fs, exponent=-2)
    t_scales, flucs, alpha = compute_fluctuations(brown, fs, method='dfa')
    assert np.isclose(alpha, 1.5, atol=0.1)

def test_compute_fluctuations_rs():

    # Note: these tests need a high sampling rate, so we use local simulations

    # Test white noise: expected RS of 0.5
    fs = 1000
    white = sim_powerlaw(N_SECONDS, fs, exponent=0)
    t_scales, flucs, hurst = compute_fluctuations(white, fs, method='rs')
    assert np.isclose(hurst, 0.5, atol=0.1)

def test_compute_rescaled_range(tsig):

    out = compute_rescaled_range(tsig, 10)
    assert isinstance(out, float)
    assert out >= 0

def test_compute_detrended_fluctuation(tsig):

    out = compute_detrended_fluctuation(tsig, 10)
    assert isinstance(out, float)
    assert out >= 0
