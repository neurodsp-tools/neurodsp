"""Tests for filter check functions."""

from pytest import raises

from neurodsp.tests.settings import FS

from neurodsp.filt.fir import design_fir_filter
from neurodsp.filt.checks import *

###################################################################################################
###################################################################################################

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

    filter_coefs = design_fir_filter(FS, 'bandpass', (8, 12))
    
    passes = check_filter_properties(filter_coefs, 1, FS, 'bandpass', (8, 12))
    assert passes is True

    filter_coefs = design_fir_filter(FS, 'bandstop', (8, 12))
    passes = check_filter_properties(filter_coefs, 1, FS, 'bandpass', (8, 12))
    assert passes is False

    filter_coefs = design_fir_filter(FS, 'bandpass', (20, 21))
    passes = check_filter_properties(filter_coefs, 1, FS, 'bandpass', (8, 12))
    assert passes is False


def test_check_filter_length():

    check_filter_length(1000, 500)
    with raises(ValueError):
        check_filter_length(500, 1000)
