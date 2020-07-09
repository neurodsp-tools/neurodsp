"""Test filtering functions."""

from pytest import raises, warns

from neurodsp.tests.settings import FS, EPS_FILT

from neurodsp.filt.filter import *
from neurodsp.filt.filter import _iir_checks

import numpy as np

###################################################################################################
###################################################################################################

def test_filter_signal(tsig, tsig_sine):

    out = filter_signal(tsig, FS, 'bandpass', (8, 12), filter_type='fir')
    assert out.shape == tsig.shape

    out = filter_signal(tsig, FS, 'bandpass', (8, 12), filter_type='iir', butterworth_order=3)
    assert out.shape == tsig.shape

    with raises(ValueError):
        out = filter_signal(tsig, FS, 'bandpass', (8, 12), filter_type='bad')

    # Apply lowpass to low-frequency sine. There should be little attenuation.
    sig_filt_lp = filter_signal(tsig_sine, FS, pass_type='lowpass', f_range=(None, 10))
    # Compare the two signals only at those times where the filtered signal is not nan.
    not_nan = ~np.isnan(sig_filt_lp)
    assert np.max(np.abs(tsig_sine[not_nan] - sig_filt_lp[not_nan])) < EPS_FILT

    # Now apply a high pass filter. The signal should be significantly attenuated.
    sig_filt_hp = filter_signal(tsig_sine, FS, pass_type='highpass', f_range=(30, None))
    not_nan = ~np.isnan(sig_filt_hp)
    assert np.max(np.abs(sig_filt_hp[not_nan])) < EPS_FILT


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
