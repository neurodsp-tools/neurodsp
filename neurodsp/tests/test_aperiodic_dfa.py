"""Tests for fractal analysis using fluctuation measures."""

from neurodsp.tests.settings import FS

from neurodsp.aperiodic.dfa import *

###################################################################################################
###################################################################################################

def test_compute_fluctuations(tsig):

    t_scales, flucs, exp = compute_fluctuations(tsig, FS)

def test_compute_rescaled_range(tsig):

    rs = compute_rescaled_range(tsig, 10)

def test_compute_detrended_fluctuation(tsig):

    out = compute_detrended_fluctuation(tsig, 10)
