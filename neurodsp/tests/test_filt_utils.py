""".  """

from pytest import raises

from neurodsp.filt.utils import *
from neurodsp.filt.fir import _fir_checks

###################################################################################################
###################################################################################################

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

def test_infer_passtype():

    assert infer_passtype((None, 1)) == 'lowpass'
    assert infer_passtype((1, None)) == 'highpass'
    assert infer_passtype((1, 2)) == 'bandpass'

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
