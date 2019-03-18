"""
test_plts_filt.py
Test filtering plots
"""

import numpy as np
from neurodsp.filt import filter_signal, design_fir_filter, compute_frequency_response
from neurodsp.plts.filt import *
from .util import plot_test


@plot_test
def test_plot_frequency_response_call():
    """
    Confirm frequency response plotting function is called and makes a plot
    """

    # Test plotting through the filter function
    sig = np.random.randn(2000)
    fs = 1000
    sig_filt = filter_signal(sig, fs, 'bandpass', (8, 12), plot_properties=True)
    assert True

@plot_test
def test_plot_filter_properties():

    kernel = design_fir_filter(2000, 1000, 'bandpass', (8, 12), 3)
    f_db, db = compute_frequency_response(kernel, a_vals=1, fs=1000)

    plot_filter_properties(f_db, db, kernel)

    assert True

@plot_test
def test_plot_frequency_response():
    """
    Confirm frequency response plotting function works directly
    """

    kernel = design_fir_filter(2000, 1000, 'bandpass', (8, 12), 3)
    f_db, db = compute_frequency_response(kernel, a_vals=1, fs=1000)

    plot_frequency_response(f_db, db)

    assert True

@plot_test
def test_plot_filter_kernel():

    kernel = design_fir_filter(2000, 1000, 'bandpass', (8, 12), 3)

    plot_filter_kernel(kernel)

    assert True
