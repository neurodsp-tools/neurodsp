"""Tests for IIR filters."""

import numpy as np

from neurodsp.tests.settings import FS

from neurodsp.filt.iir import *

###################################################################################################
###################################################################################################

def test_filter_signal_iir(tsig):

    out = filter_signal_iir(tsig, FS, 'bandpass', (8, 12), 3)
    assert out.shape == tsig.shape

def test_filter_signal_iir_2d(tsig2d):

   out = filter_signal_iir(tsig2d, FS, 'bandpass', (8, 12), 3)
   assert out.shape == tsig2d.shape
   assert sum(~np.isnan(out[0, :])) > 0

   out, sos = filter_signal_iir(tsig2d, FS, 'bandpass', (8, 12), 3, return_filter=True)
   assert np.shape(sos)[1] == 6

def test_apply_iir_filter(tsig):

    out = apply_iir_filter(tsig, np.array([1, 1, 1, 1, 1, 1]))
    assert out.shape == tsig.shape

def test_design_iir_filter():

    sig_length, butter_order = 1000, 3
    test_filts = {'bandpass' : (5, 10), 'bandstop' : (5, 6),
                  'lowpass' : (None, 5), 'highpass' : (5, None)}

    for pass_type, f_range in test_filts.items():
        filter_coefs = design_iir_filter(sig_length, pass_type, f_range, butter_order)
