"""Tests for fractal analysis using fluctuation measures."""

from pytest import raises

import numpy as np

from neurodsp.sim import sim_powerlaw
from neurodsp.tests.settings import FS, FS_HIGH

from neurodsp.aperiodic.dfa import (compute_fluctuations, compute_rescaled_range,
                                    compute_detrended_fluctuation)

###################################################################################################
###################################################################################################

def test_compute_fluctuations(tsig):

    t_scales, flucs, exp = compute_fluctuations(tsig, 500)
    assert len(t_scales) == len(flucs)

    # Check error for if the settings create window lengths that are too short
    with raises(ValueError):
        t_scales, flucs, exp = compute_fluctuations(tsig, 100)

    # Check error for nonsense method input
    with raises(ValueError):
        t_scales, flucs, exp = compute_fluctuations(tsig, 500, method='nope.')

def test_compute_fluctuations_dfa(tsig_white, tsig_brown):
    """Tests fluctuation analysis for the DFA method."""

    # Test white noise: expected DFA of 0.5
    t_scales, flucs, exp = compute_fluctuations(tsig_white, FS_HIGH, method='dfa')
    assert np.isclose(exp, 0.5, atol=0.1)

    # Test brown noise: expected DFA of 1.5
    t_scales, flucs, exp = compute_fluctuations(tsig_brown, FS_HIGH, method='dfa')
    assert np.isclose(exp, 1.5, atol=0.1)

def test_compute_fluctuations_rs(tsig_white):
    """Tests fluctuation analysis for the RS method."""

    # Test white noise: expected RS of 0.5
    t_scales, flucs, exp = compute_fluctuations(tsig_white, FS_HIGH, method='rs')
    assert np.isclose(exp, 0.5, atol=0.1)

def test_compute_rescaled_range(tsig):

    out = compute_rescaled_range(tsig, 10)
    assert isinstance(out, float)
    assert out >= 0

def test_compute_detrended_fluctuation(tsig):

    out = compute_detrended_fluctuation(tsig, 10)
    assert isinstance(out, float)
    assert out >= 0
