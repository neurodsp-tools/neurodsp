"""Tests for neurodsp.aperiodic.conversions."""

from neurodsp.aperiodic.conversions import *

###################################################################################################
###################################################################################################

def test_convert_exponent_alpha():

    exp = 1.5
    alpha = convert_exponent_alpha(exp)
    assert isinstance(alpha, float)

def test_convert_alpha_exponent():

    alpha = 1.5
    exp = convert_exponent_alpha(alpha)
    assert isinstance(exp, float)

def test_convert_exponent_hurst():

    exp = 1.5

    hurst1 = convert_exponent_hurst(exp, 'gaussian')
    assert isinstance(hurst1, float)

    hurst2 = convert_exponent_hurst(exp, 'brownian')
    assert isinstance(hurst2, float)

def test_convert_hurst_exponent():

    hurst = 1.5

    exp1 = convert_hurst_exponent(hurst, 'gaussian')
    assert isinstance(exp1, float)

    exp2 = convert_hurst_exponent(hurst, 'brownian')
    assert isinstance(exp2, float)

def test_convert_exponent_hfd():

    exp = 1.5
    hfd = convert_exponent_hfd(exp)
    assert isinstance(hfd, float)

def test_convert_hfd_exponent():

    hfd = 1.5
    exp = convert_exponent_alpha(hfd)
    assert isinstance(exp, float)
