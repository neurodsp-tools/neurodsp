"""Tests for neurodsp.filt.utils."""

import tempfile
from pytest import raises, mark, param

import numpy as np

from neurodsp.tests.settings import FS
from neurodsp.filt.utils import *
from neurodsp.filt.fir import design_fir_filter, compute_filter_length
from neurodsp.filt.iir import design_iir_filter
from neurodsp.filt.checks import check_filter_definition, check_filter_properties

from neurodsp.filt.utils import *

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

    # Get the length for a possible filter & calculate # of values that should be dropped for it
    sig_len = 1000
    sig = np.ones(sig_len)
    filt_len = compute_filter_length(FS, 'bandpass', f_lo=4, f_hi=8, n_cycles=3, n_seconds=None)
    n_rmv = int(np.ceil(filt_len / 2))

    dropped_sig = remove_filter_edges(sig, filt_len)

    assert np.all(np.isnan(dropped_sig[:n_rmv]))
    assert np.all(np.isnan(dropped_sig[-n_rmv:]))
    assert np.all(~np.isnan(dropped_sig[n_rmv:-n_rmv]))


@mark.parametrize("pass_type", ['bandpass', 'bandstop', 'lowpass', 'highpass'])
@mark.parametrize("filt_type", ['IIR', 'FIR'])
def test_gen_filt_str(pass_type, filt_type):

    f_db = np.arange(0, 50)
    db = np.random.rand(50)
    pass_bw = 10
    transition_bw = 4
    f_range = (10, 40)
    f_range_trans = (40, 44)
    order = 1

    report_str = gen_filt_str(pass_type, filt_type, FS, f_db, db, pass_bw,
                              transition_bw, f_range, f_range_trans, order)

    assert pass_type in report_str
    assert filt_type in report_str


@mark.parametrize("dir_exists", [True, param(False, marks=mark.xfail)])
@mark.parametrize("filt_type", ['IIR', 'FIR'])
def test_save_filt_report(dir_exists, filt_type):

    pass_type = 'bandpass'
    f_range = (10, 40)

    f_db = np.arange(1, 100)
    db = np.random.rand(99)

    pass_bw = 10
    transition_bw = 4
    f_range_trans = (40, 44)

    order = 1

    if pass_type == 'FIR':
        filter_coefs = np.random.rand(10)
    else:
        filter_coefs = None

    temp_path = tempfile.NamedTemporaryFile()

    if not dir_exists:
        save_filt_report('/bad/path/', pass_type, filt_type, FS, f_db, db,  pass_bw,
                         transition_bw, f_range, f_range_trans, order, filter_coefs=filter_coefs)
    else:
        print(temp_path.name)
        save_filt_report(temp_path.name, pass_type, filt_type, FS, f_db, db,  pass_bw,
                         transition_bw, f_range, f_range_trans, order, filter_coefs=filter_coefs)

    temp_path.close()
