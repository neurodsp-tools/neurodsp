"""Test filtering functions."""

from pytest import raises, warns

import numpy as np

from neurodsp.filt import *
from neurodsp.filt import _fir_checks, _iir_checks

###################################################################################################
###################################################################################################

def test_filter_signal():
    pass

def test_filter_signal_fir(tsig):

    sig = filter_signal_fir(tsig, 500, 'bandpass', (8, 12))
    assert True

def test_filter_signal_iir(tsig):

    sig = filter_signal_iir(tsig, 500, 'bandpass', (8, 12), 3)
    assert True

def test_design_fir_filter():
    pass

def test_design_iir_filter():
    pass

def test_check_filter_definition():

    # Check that error catching works for bad pass_type definition
    with raises(ValueError):
        check_filter_definition('not_a_filter', (12, 12))

    # Check that filter definitions that are legal evaluate as expected
    #   Float or partially filled tuple for f_range should work passable
    f_lo, f_hi = check_filter_definition('bandpass', f_range=(8, 12))
    assert f_lo == 8; assert f_hi == 12
    f_lo, f_hi = check_filter_definition('bandstop', f_range=(58, 62))
    assert f_lo == 58; assert f_hi == 62
    f_lo, f_hi = check_filter_definition('lowpass', f_range=58)
    assert f_lo == None ; assert f_hi == 58
    f_lo, f_hi = check_filter_definition('lowpass', f_range=(0, 58))
    assert f_lo == None ; assert f_hi == 58
    f_lo, f_hi = check_filter_definition('highpass', f_range=58)
    assert f_lo == 58 ; assert f_hi == None
    f_lo, f_hi = check_filter_definition('highpass', f_range=(58, 100))
    assert f_lo == 58 ; assert f_hi == None

    # Check that a bandpass & bandstop definitions fail without proper definitions
    with raises(ValueError):
        f_lo, f_hi = check_filter_definition('bandpass', f_range=8)
    with raises(ValueError):
        f_lo, f_hi = check_filter_definition('bandstop', f_range=58)

    # Check that frequencies cannot be inverted
    with raises(ValueError):
        f_lo, f_hi = check_filter_definition('bandpass', f_range=(8, 4))
    with raises(ValueError):
        f_lo, f_hi = check_filter_definition('bandstop', f_range=(62, 58))

def test_check_filter_properties():
    pass

def compute_frequency_response():
    pass

def compute_pass_band():

    fs = 500
    assert compute_pass_band(fs, 'bandpass', (4, 8)) == 4.
    assert compute_pass_band(fs, 'highpass', 20) == 20.
    assert compute_pass_band(fs, 'lowpass', 5) == compute_nyquist(fs) - 5

def compute_transition_band():
    pass

def test_compute_nyquist():

    assert compute_nyquist(100.) == 50.
    assert compute_nyquist(256) == 128.

def test_infer_passtype():

    assert infer_passtype((None, 1)) == 'lowpass'
    assert infer_passtype((1, None)) == 'highpass'
    assert infer_passtype((1, 2)) == 'bandpass'

def test_remove_filter_edges():

    # Get the length for a possible filter & calc # of values should be dropped for it
    sig_len = 1000
    sig = np.ones([1, sig_len])
    filt_len = _fir_checks('bandpass', 4, 8, 3, None, 500, sig_len)
    n_rmv = int(np.ceil(filt_len / 2))

    dropped_sig = remove_filter_edges(sig, filt_len)

    assert np.all(np.isnan(dropped_sig[:n_rmv]))
    assert np.all(np.isnan(dropped_sig[-n_rmv:]))
    assert np.all(~np.isnan(dropped_sig[n_rmv:-n_rmv]))

###################################################################################################
################################### TEST FILT PRIVATE FUNCTIONS ###################################
###################################################################################################

def test_fir_checks():

    # Settings for checks
    fs = 500
    sig_length = 2000
    f_lo, f_hi = 4, 8

    # Check filt_len, if defined using n_seconds
    n_seconds = 1.75 # Number chosen to create odd expected filt_len (not needing rounding up)
    expected_filt_len = n_seconds * fs
    filt_len = _fir_checks('bandpass', f_lo, f_hi, n_cycles=None, n_seconds=n_seconds,
                           fs=fs, sig_length=sig_length)
    assert filt_len == expected_filt_len

    # Check filt_len, if defined using n_cycles
    n_cycles = 5
    expected_filt_len = int(np.ceil(fs * n_cycles / f_lo))
    filt_len = _fir_checks('bandpass', f_lo, f_hi, n_cycles=n_cycles, n_seconds=None,
                           fs=fs, sig_length=sig_length)
    assert filt_len == expected_filt_len

    # Check filt_len, if expected to be rounded up to be odd
    n_cycles = 4
    expected_filt_len = int(np.ceil(fs * n_cycles / f_lo)) + 1
    filt_len = _fir_checks('bandpass', f_lo, f_hi, n_cycles=n_cycles, n_seconds=None,
                           fs=fs, sig_length=sig_length)
    assert filt_len == expected_filt_len

    # Check error is raised if the filter designed is longer than the signal
    with raises(ValueError):
        _fir_checks('bandpass', f_lo, f_hi, None, 3, fs, sig_length=1000)

def test_iir_checks():

    # Check catch for having n_seconds defined
    with raises(ValueError):
        _iir_checks(1, 3, None)
    # Check catch for not having butterworth_order defined
    with raises(ValueError):
        _iir_checks(None, None, None)
    # Check catch for having remove_edges defined
    with warns(UserWarning):
        _iir_checks(None, 3, True)
