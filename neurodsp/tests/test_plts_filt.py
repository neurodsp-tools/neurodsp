"""Test filtering plots."""

from neurodsp.filt.filter import filter_signal
from neurodsp.filt.fir import design_fir_filter
from neurodsp.filt.utils import compute_frequency_response

from neurodsp.tests.settings import FS
from neurodsp.tests.utils import plot_test

from neurodsp.plts.filt import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_filter_properties():

    coefs = design_fir_filter(FS, 'bandpass', (8, 12), 3)
    f_db, db = compute_frequency_response(coefs, a_vals=1, fs=FS)

    plot_filter_properties(f_db, db, FS, coefs)

@plot_test
def test_plot_frequency_response():

    coefs = design_fir_filter(FS, 'bandpass', (8, 12), 3)
    f_db, db = compute_frequency_response(coefs, a_vals=1, fs=FS)

    plot_frequency_response(f_db, db)

@plot_test
def test_plot_impulse_response():

    coefs = design_fir_filter(FS, 'bandpass', (8, 12), 3)

    plot_impulse_response(FS, coefs)
