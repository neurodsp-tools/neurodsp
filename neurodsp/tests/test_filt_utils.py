"""Tests for filter utilities."""

from pytest import raises
from neurodsp.tests.settings import FS

from neurodsp.filt.utils import *
from neurodsp.filt.fir import design_fir_filter, compute_filter_length

###################################################################################################
###################################################################################################

def test_infer_passtype():

    assert infer_passtype((None, 1)) == 'lowpass'
    assert infer_passtype((1, None)) == 'highpass'
    assert infer_passtype((1, 2)) == 'bandpass'

def test_compute_frequency_response():

    filter_coefs = design_fir_filter(FS, 'bandpass', (8, 12))
    f_db, db = compute_frequency_response(filter_coefs, 1, FS)

    with raises(ValueError):
        f_db, db = compute_frequency_response(filter_coefs, None, FS)

def test_compute_pass_band():

    assert compute_pass_band(FS, 'bandpass', (4, 8)) == 4.
    assert compute_pass_band(FS, 'lowpass', 20) == 20.
    assert compute_pass_band(FS, 'highpass', 5) == compute_nyquist(FS) - 5

def test_compute_transition_band():

    filter_coefs = design_fir_filter(FS, 'bandpass', (8, 12))
    f_db, db = compute_frequency_response(filter_coefs, 1, FS)
    trans_band = compute_transition_band(f_db, db)

def test_compute_nyquist():

    assert compute_nyquist(100.) == 50.
    assert compute_nyquist(256) == 128.

def test_remove_filter_edges():

    # Get the length for a possible filter & calculate # of values should be dropped for it
    sig_len = 1000
    sig = np.ones(sig_len)
    filt_len = compute_filter_length(FS, 'bandpass', f_lo=4, f_hi=8, n_cycles=3, n_seconds=None)
    n_rmv = int(np.ceil(filt_len / 2))

    dropped_sig = remove_filter_edges(sig, filt_len)

    assert np.all(np.isnan(dropped_sig[:n_rmv]))
    assert np.all(np.isnan(dropped_sig[-n_rmv:]))
    assert np.all(~np.isnan(dropped_sig[n_rmv:-n_rmv]))
