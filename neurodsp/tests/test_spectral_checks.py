"""Test spectral checker functions."""

import numpy as np

from neurodsp.spectral.checks import *

###################################################################################################
###################################################################################################

def test_spg_settings():

    fs = 500
    nperseg, noverlap = check_spg_settings(fs, 'hann', None, None)
    assert nperseg == fs
    assert noverlap == None

    window = np.array([1, 2, 3, 4])
    noverlap = 20
    nperseg, noverlap = check_spg_settings(fs, window, None, noverlap)
    assert nperseg == len(window)
    assert noverlap == noverlap
