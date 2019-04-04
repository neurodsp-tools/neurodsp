"""Tests for IIR filters."""

from neurodsp.filt.iir import *

###################################################################################################
###################################################################################################

def test_filter_signal_iir(tsig):

    sig = filter_signal_iir(tsig, 500, 'bandpass', (8, 12), 3)
    assert True

def test_design_iir_filter():

    sig_length, butter_order = 1000, 3
    test_filts = {'bandpass' : (5, 10), 'bandstop' : (5, 6),
                  'lowpass' : (None, 5), 'highpass' : (5, None)}

    for pass_type, f_range in test_filts.items():
        filter_coefs = design_iir_filter(sig_length, pass_type, f_range, butter_order)
    assert True
