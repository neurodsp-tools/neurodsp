"""Tests for neurodsp.spectral.utils."""

import pytest

import numpy as np
from numpy.testing import assert_equal

from neurodsp.tests.settings import FS
from neurodsp.spectral.utils import *

###################################################################################################
###################################################################################################

def test_trim_spectrum():

    freqs = np.array([5, 6, 7, 8, 9])
    pows = np.array([1, 2, 3, 4, 5])

    freqs_new, pows_new = trim_spectrum(freqs, pows, [6, 8])
    assert_equal(freqs_new, np.array([6, 7, 8]))
    assert_equal(pows_new, np.array([2, 3, 4]))

def test_trim_spectrogram():

    freqs = np.array([5, 6, 7, 8])
    times = np.array([0, 1, 2,])
    pows = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9],
                     [10, 11, 12]])

    f_ext, t_ext, p_ext = trim_spectrogram(freqs, times, pows, f_range=[6, 8], t_range=[0, 1])
    assert_equal(f_ext, np.array([6, 7, 8]))
    assert_equal(t_ext, np.array([0, 1]))
    assert_equal(p_ext, np.array([[4, 5], [7, 8], [10, 11]]))

    # Check extraction across specified axis
    f_ext, t_ext, p_ext = trim_spectrogram(freqs, times, pows, f_range=None, t_range=[0, 1])
    assert_equal(f_ext, freqs)
    assert_equal(t_ext, np.array([0, 1]))
    f_ext, t_ext, p_ext = trim_spectrogram(freqs, times, pows, f_range=[6, 8], t_range=None)
    assert_equal(f_ext, np.array([6, 7, 8]))
    assert_equal(t_ext, times)


@pytest.mark.parametrize("fast_len", [True, False])
def test_window_pad(fast_len):

    nperseg = 100
    noverlap = 10
    npad = 1000

    sig = np.random.rand(1000)

    sig_windowed, _nperseg, _noverlap = window_pad(sig, nperseg, noverlap, npad, fast_len)

    # Overlap was handled correctly b/w the first two windows
    assert np.all(sig_windowed[npad:npad+nperseg][-noverlap:] ==
        sig_windowed[(3*npad)+nperseg:(3*npad)+nperseg+noverlap])

    # Updated nperseg has no remainder
    nwin = (len(sig_windowed) / nperseg)
    assert nwin == int(nwin)

    # Ensure updated nperseg is correct
    assert _nperseg == nperseg + (2*npad)
