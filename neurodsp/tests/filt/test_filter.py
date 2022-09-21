"""Tests for neurodsp.filt.filter."""

from pytest import raises, warns

import numpy as np

from neurodsp.tests.settings import FS, F_RANGE

from neurodsp.filt.filter import *
from neurodsp.filt.filter import _iir_checks

###################################################################################################
###################################################################################################

def test_filter_signal(tsig):

    out = filter_signal(tsig, FS, 'bandpass', F_RANGE, filter_type='fir')
    assert out.shape == tsig.shape

    out = filter_signal(tsig, FS, 'bandpass', F_RANGE, filter_type='iir', butterworth_order=3)
    assert out.shape == tsig.shape

    # 2d check
    sigs = np.array([tsig, tsig])
    outs = filter_signal(sigs, FS, 'bandpass', F_RANGE, remove_edges=False)
    assert np.diff(outs, axis=0).sum() == 0

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
