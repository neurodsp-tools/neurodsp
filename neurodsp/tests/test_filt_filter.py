"""Test filtering functions."""

from pytest import raises, warns

from neurodsp.filt.filter import *
from neurodsp.filt.filter import _iir_checks

###################################################################################################
###################################################################################################

def test_filter_signal(tsig):

    out = filter_signal(tsig, 500, 'bandpass', (8, 12), filt_type='fir')
    out = filter_signal(tsig, 500, 'bandpass', (8, 12), filt_type='iir', butterworth_order=3)
    with raises(ValueError):
        out = filter_signal(tsig, 500, 'bandpass', (8, 12), filt_type='bad')
    assert True

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
