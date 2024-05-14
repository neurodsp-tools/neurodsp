"""Tests for neurodsp.filt.filter."""

from pytest import raises, warns

import numpy as np

from neurodsp.tests.settings import FS, F_RANGE

from neurodsp.filt.filter import *
from neurodsp.filt.filter import _filter_input_checks

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

def test_filter_input_checks():

    fir_inputs = {'n_cycles' : 5, 'remove_edges' : False}
    _filter_input_checks('fir', fir_inputs)

    iir_inputs = {'butterworth_order' : 7}
    _filter_input_checks('iir', iir_inputs)

    mixed_inputs = {'n_cycles' : 5, 'butterworth_order' : 7}
    extra_inputs = {'n_cycles' : 5, 'nonsense_input' : True}

    with raises(AssertionError):
        _filter_input_checks('fir', iir_inputs)
    with raises(AssertionError):
        _filter_input_checks('iir', fir_inputs)
    with raises(AssertionError):
        _filter_input_checks('fir', mixed_inputs)
    with raises(AssertionError):
        _filter_input_checks('fir', mixed_inputs)
    with raises(AssertionError):
        _filter_input_checks('fir', extra_inputs)
