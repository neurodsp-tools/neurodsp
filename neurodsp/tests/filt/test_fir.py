"""Tests for neurodsp.filt.fir."""

from pytest import raises

import numpy as np

from neurodsp.tests.tsettings import FS, EPS_FILT

from neurodsp.filt.fir import *

###################################################################################################
###################################################################################################

def test_filter_signal_fir(tsig, tsig_sine):

    out = filter_signal_fir(tsig, FS, 'bandpass', (8, 12))
    assert out.shape == tsig.shape

    # Apply lowpass to low-frequency sine, which should should give little attenuation
    sig_filt_lp = filter_signal_fir(tsig_sine, FS, pass_type='lowpass', f_range=(None, 10))

    # Compare the two signals at timepoints where the filtered signal is not nan
    not_nan = ~np.isnan(sig_filt_lp)
    assert np.allclose(tsig_sine[not_nan], sig_filt_lp[not_nan], atol=EPS_FILT)

    # Apply a highpass filter, which should significantly attenuate the signal
    sig_filt_hp = filter_signal_fir(tsig_sine, FS, pass_type='highpass', f_range=(30, None))

    # Compare the two signals at timepoints where the filtered signal is not nan
    not_nan = ~np.isnan(sig_filt_hp)
    sig_filt_hp = sig_filt_hp[not_nan]
    expected_answer = np.zeros_like(sig_filt_hp)
    assert np.allclose(sig_filt_hp, expected_answer, atol=EPS_FILT)

def test_filter_signal_fir_2d(tsig2d):

    out = filter_signal_fir(tsig2d, FS, 'bandpass', (8, 12))
    assert out.shape == tsig2d.shape
    assert sum(~np.isnan(out[0, :])) > 0

def test_design_fir_filter():

    test_filts = {'bandpass' : (5, 10), 'bandstop' : (5, 6),
                  'lowpass' : (None, 5), 'highpass' : (5, None)}

    for pass_type, f_range in test_filts.items():
        filter_coefs = design_fir_filter(FS, pass_type, f_range, n_cycles=3)

def test_apply_fir_filter(tsig):

    out = apply_fir_filter(tsig, np.array([1, 1, 1, 1, 1]))
    assert out.shape == tsig.shape

def test_compute_filter_length():

    # Settings for checks
    fs = 500
    f_lo, f_hi = 4, 8

    # Check filt_len, if defined using n_seconds
    #   n_seconds here is chosen to create expected odd filt_len, without needing rounding up
    n_seconds = 1.75
    expected_filt_len = n_seconds * fs
    filt_len = compute_filter_length(fs, 'bandpass', f_lo, f_hi, n_seconds=n_seconds)
    assert filt_len == expected_filt_len

    # Check filt_len, if defined using n_cycles
    n_cycles = 5
    expected_filt_len = int(np.ceil(fs * n_cycles / f_lo))
    filt_len = compute_filter_length(fs, 'bandpass', f_lo, f_hi, n_cycles=n_cycles)
    assert filt_len == expected_filt_len

    # Check filt_len, if expected to be rounded up to be odd
    n_cycles = 4
    expected_filt_len = int(np.ceil(fs * n_cycles / f_lo)) + 1
    filt_len = compute_filter_length(fs, 'bandpass', f_lo, f_hi, n_cycles=n_cycles)
    assert filt_len == expected_filt_len
    with raises(ValueError):
        filt_len = compute_filter_length(fs, 'bandpass', f_lo, f_hi)

    # Test error with inconsistent inputs
    with raises(ValueError):
        compute_filter_length(fs, 'bandpass', f_lo, f_hi, n_cycles=3, n_seconds=2.0)
