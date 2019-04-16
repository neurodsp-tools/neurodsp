"""Tests for timefrequency estimations using wavelets."""

from neurodsp.timefrequency.wavelets import *

###################################################################################################
###################################################################################################

def test_morlet_transform(tsig):

    out = morlet_transform(tsig, 500, [5, 10, 15])
    assert True

def test_morlet_convolve(tsig):

    out = morlet_convolve(tsig, 500, 10)
    #out = morlet_convolve(tsig, 500, 10, norm='amp')
    assert True
