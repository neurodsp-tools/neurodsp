"""Tests for neurodsp.filt.checks."""

import warnings

from pytest import raises

from neurodsp.tests.settings import FS

from neurodsp.filt.fir import design_fir_filter
from neurodsp.filt.checks import *

###################################################################################################
###################################################################################################

def test_check_filter_definition():

    # Check error for bad pass_type definition
    with raises(ValueError):
        check_filter_definition('not_a_filter', (12, 12))

    # Check that filter definitions that are legal evaluate as expected
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

    # Check that bandpass & bandstop definitions fail without proper definitions
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

    filter_coefs = design_fir_filter(FS, 'bandpass', (8, 12))

    # Check valid / passing filter
    passes = check_filter_properties(filter_coefs, 1, FS, 'bandpass', (8, 12))
    assert passes is True

    # Check failing filter - insufficient attenuation
    with warnings.catch_warnings(record=True) as warn:
        filter_coefs = design_fir_filter(FS, 'bandstop', (8, 12))
        passes = check_filter_properties(filter_coefs, 1, FS, 'bandpass', (8, 12))
    assert passes is False
    assert len(warn) == 1
    assert "filter attenuation" in str(warn[-1].message)

    # Check failing filter - transition bandwidth
    with warnings.catch_warnings(record=True) as warn:
        filter_coefs = design_fir_filter(FS, 'bandpass', (20, 21))
        passes = check_filter_properties(filter_coefs, 1, FS, 'bandpass', (8, 12))
    assert passes is False
    assert len(warn) == 1
    assert "Transition bandwidth" in str(warn[-1].message)

    # Check failing filter - insufficient attenuation on part of stopband
    with warnings.catch_warnings(record=True) as w:
        filter_coefs = design_fir_filter(FS, 'bandpass', (5, 40), n_cycles=1)
        check_filter_properties(filter_coefs, 1, FS, 'bandpass', (5, 40))
    assert len(w) == 1
    assert "stopband" in str(w[-1].message)

def test_check_filter_length():

    check_filter_length(1000, 500)
    with raises(ValueError):
        check_filter_length(500, 1000)
