"""Tests neurodsp.spectral.checks."""

import numpy as np

from neurodsp.tests.tsettings import FS

from neurodsp.spectral.checks import *

###################################################################################################
###################################################################################################

def test_check_windowing_settings():

    nperseg, noverlap = check_windowing_settings(FS, 'hann', None, None)
    assert nperseg == FS
    assert noverlap == None

    window = np.array([1, 2, 3, 4])
    noverlap_in = 20
    nperseg, noverlap_out = check_windowing_settings(FS, window, None, noverlap_in)
    assert nperseg == len(window)
    assert noverlap_out == noverlap_in

    # Check next fast len
    nperseg, noverlap = check_windowing_settings(FS, 'hann', 101, None, fast_len=True)
    assert nperseg == 105
