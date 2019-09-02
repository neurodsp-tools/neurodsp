"""Tests for FIR filters."""

import numpy as np

from neurodsp.tests.settings import FS

from neurodsp.filt.fir import *

###################################################################################################
###################################################################################################

def test_filter_signal_fir(tsig):

    out = filter_signal_fir(tsig, FS, 'bandpass', (8, 12))
    assert out.shape == tsig.shape

def test_filter_signal_fir_2d(tsig2d):

    out = filter_signal_fir(tsig2d, FS, 'bandpass', (8, 12))
    assert out.shape == tsig2d.shape
    assert sum(~np.isnan(out[0, :])) > 0

def test_design_fir_filter():

    test_filts = {'bandpass' : (5, 10), 'bandstop' : (5, 6),
                  'lowpass' : (None, 5), 'highpass' : (5, None)}

    for pass_type, f_range in test_filts.items():
        filter_coefs = design_fir_filter(FS, pass_type, f_range)

def test_apply_fir_filter(tsig):

    out = apply_fir_filter(tsig, np.array([1, 1, 1, 1, 1]))
    assert out.shape == tsig.shape

def test_compute_filter_length():

    # Settings for checks
    fs = 500
    f_lo, f_hi = 4, 8

    # Check filt_len, if defined using n_seconds
    n_seconds = 1.75 # Number chosen to create odd expected filt_len (not needing rounding up)
    expected_filt_len = n_seconds * fs
    filt_len = compute_filter_length(fs, 'bandpass', f_lo, f_hi, n_cycles=None, n_seconds=n_seconds)
    assert filt_len == expected_filt_len

    # Check filt_len, if defined using n_cycles
    n_cycles = 5
    expected_filt_len = int(np.ceil(fs * n_cycles / f_lo))
    filt_len = compute_filter_length(fs, 'bandpass', f_lo, f_hi, n_cycles=n_cycles, n_seconds=None)
    assert filt_len == expected_filt_len

    # Check filt_len, if expected to be rounded up to be odd
    n_cycles = 4
    expected_filt_len = int(np.ceil(fs * n_cycles / f_lo)) + 1
    filt_len = compute_filter_length(fs, 'bandpass', f_lo, f_hi, n_cycles=n_cycles, n_seconds=None)
    assert filt_len == expected_filt_len
