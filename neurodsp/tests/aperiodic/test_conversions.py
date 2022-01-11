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
