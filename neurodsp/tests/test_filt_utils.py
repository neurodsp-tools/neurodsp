"""Tests for filter utilities."""

from neurodsp.filt.utils import *
from neurodsp.filt.fir import design_fir_filter, compute_filt_len

###################################################################################################
###################################################################################################

def test_infer_passtype():

    assert infer_passtype((None, 1)) == 'lowpass'
    assert infer_passtype((1, None)) == 'highpass'
    assert infer_passtype((1, 2)) == 'bandpass'

def test_compute_frequency_response():

    filter_coefs = design_fir_filter(1000, 500, 'bandpass', (8, 12))
    f_db, db = compute_frequency_response(filter_coefs, 1, 500)
    assert True

def test_compute_pass_band():

    fs = 500
    assert compute_pass_band(fs, 'bandpass', (4, 8)) == 4.
    assert compute_pass_band(fs, 'lowpass', 20) == 20.
    assert compute_pass_band(fs, 'highpass', 5) == compute_nyquist(fs) - 5

def test_compute_transition_band():

    filter_coefs = design_fir_filter(1000, 500, 'bandpass', (8, 12))
    f_db, db = compute_frequency_response(filter_coefs, 1, 500)
    trans_band = compute_transition_band(f_db, db)
    assert True

def test_compute_nyquist():

    assert compute_nyquist(100.) == 50.
    assert compute_nyquist(256) == 128.

def test_remove_filter_edges():

    # Get the length for a possible filter & calc # of values should be dropped for it
    sig_len = 1000
    sig = np.ones([1, sig_len])
    filt_len = compute_filt_len(sig_len, 500, 'bandpass', 4, 8, 3, None)
    n_rmv = int(np.ceil(filt_len / 2))

    dropped_sig = remove_filter_edges(sig, filt_len)

    assert np.all(np.isnan(dropped_sig[:n_rmv]))
    assert np.all(np.isnan(dropped_sig[-n_rmv:]))
    assert np.all(~np.isnan(dropped_sig[n_rmv:-n_rmv]))
