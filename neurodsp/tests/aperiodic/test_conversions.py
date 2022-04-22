"""Tests for neurodsp.aperiodic.conversions."""

import numpy as np

from neurodsp.aperiodic.conversions import *

###################################################################################################
###################################################################################################

def test_convert_exponent_alpha():

    # Check expected DFA exponent for pink noise of 1
    alpha = convert_exponent_alpha(-1)
    assert isinstance(alpha, float)
    assert np.isclose(alpha, 1)

    # Check expected DFA exponent for white noise of 0.5
    assert np.isclose(convert_exponent_alpha(0), 0.5)

def test_convert_alpha_exponent():

    # Check expected exponent for DFA exponent that should correspond to pink noise
    exp = convert_alpha_exponent(1)
    assert isinstance(exp, float)
    assert np.isclose(exp, -1)

    # Check expected exponent for DFA exponent that should correspond to pink noise
    assert np.isclose(convert_alpha_exponent(0.5), 0)

def test_convert_exponent_hurst():

    # Check expected hurst exponent for pink noise, fractional Gaussian noise
    hurst = convert_exponent_hurst(-1, 'gaussian')
    assert isinstance(hurst, float)
    assert np.isclose(hurst, 1)

    # Check expected hurst exponent for pink noise, fractional Brownian motion
    hurst = convert_exponent_hurst(-1, 'brownian')
    assert isinstance(hurst, float)
    assert np.isclose(hurst, 0)

def test_convert_hurst_exponent():

    # Check expected aperiodic exponent for pink noise hurst exponent, fractional Gaussian noise
    exp = convert_hurst_exponent(1, 'gaussian')
    assert isinstance(exp, float)
    assert np.isclose(exp, -1)

    # Check expected aperiodic exponent for pink noise hurst exponent, fractional Brownian motion
    exp = convert_hurst_exponent(0, 'brownian')
    assert isinstance(exp, float)
    assert np.isclose(exp, -1)

def test_convert_exponent_hfd():

    # Check special case of exponent 0
    assert convert_exponent_hfd(0) == 2.

    # Check expected HFD for pink noise
    hfd = convert_exponent_hfd(-1)
    assert isinstance(hfd, float)
    assert np.isclose(hfd, 2)

    # Check expected HFD at the limit
    hfd = convert_exponent_hfd(-3)
    assert isinstance(hfd, float)
    assert np.isclose(hfd, 1)

def test_convert_hfd_exponent():

    # Check expected exponent for HFD that should correspond to pink noise
    exp = convert_hfd_exponent(2)
    assert isinstance(exp, float)
    assert np.isclose(exp, -1)

    # Check expected HFD at the limit
    exp = convert_hfd_exponent(1)
    assert isinstance(exp, float)
    assert np.isclose(exp, -3)
