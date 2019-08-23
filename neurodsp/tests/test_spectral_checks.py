"""Test spectral checker functions."""

import numpy as np

from neurodsp.tests.settings import FS

from neurodsp.spectral.checks import *

###################################################################################################
###################################################################################################

def test_spg_settings():

    nperseg, noverlap = check_spg_settings(FS, 'hann', None, None)
    assert nperseg == FS
    assert noverlap == None

    window = np.array([1, 2, 3, 4])
    noverlap_in = 20
    nperseg, noverlap_out = check_spg_settings(FS, window, None, noverlap_in)
    assert nperseg == len(window)
    assert noverlap_out == noverlap_in
