"""Tests for neurodsp.spectral.measures."""

import numpy as np

from neurodsp.spectral.measures import *

###################################################################################################
###################################################################################################

def test_compute_absolute_power(tspectrum):

    abs_power = compute_absolute_power(tspectrum['freqs'], tspectrum['powers'], [8, 12])
    assert isinstance(abs_power, float)

def test_compute_relative_power(tspectrum):

    rel_power = compute_relative_power(tspectrum['freqs'], tspectrum['powers'], [8, 12])
    assert isinstance(rel_power, float)

def test_compute_band_ratio(tspectrum):

    ratio = compute_band_ratio(tspectrum['freqs'], tspectrum['powers'], [4, 8], [13, 25])
    assert isinstance(ratio, float)
