"""Tests for neurodsp.plts.spectral."""

import numpy as np

from neurodsp.spectral.power import compute_spectrum
from neurodsp.spectral.variance import compute_spectral_hist, compute_scv, compute_scv_rs

from neurodsp.tests.settings import TEST_PLOTS_PATH, FS
from neurodsp.tests.tutils import plot_test

from neurodsp.plts.spectral import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_power_spectra(tsig_comb, tsig_burst):

    freqs1, powers1 = compute_spectrum(tsig_comb, FS)
    freqs2, powers2 = compute_spectrum(tsig_burst, FS)

    plot_power_spectra(freqs1, powers1,
                       save_fig=True, file_path=TEST_PLOTS_PATH,
                       file_name='test_plot_power_spectra-1.png')

    plot_power_spectra([freqs1, freqs2], [powers1, powers2],
                       labels=['first', 'second'], colors=['k', 'r'],
                       save_fig=True, file_path=TEST_PLOTS_PATH,
                       file_name='test_plot_power_spectra-2.png')

@plot_test
def test_plot_scv(tsig_comb):

    freqs, scv = compute_scv(tsig_comb, FS)

    plot_scv(freqs, scv,
             save_fig=True, file_path=TEST_PLOTS_PATH,
             file_name='test_plot_scv.png')

def test_other_scv_plots(tsig_comb):

    freqs, t_inds, scv_rs = compute_scv_rs(tsig_comb, FS, nperseg=FS/2, method='rolling')

    _plot_scv_rs_lines(freqs, scv_rs)
    _plot_scv_rs_matrix(freqs, t_inds, scv_rs)

@plot_test
def _plot_scv_rs_lines(freqs, scv_rs):

    plot_scv_rs_lines(freqs, scv_rs,
                      save_fig=True, file_path=TEST_PLOTS_PATH,
                      file_name='test_plot_scv_rs_lines.png')

@plot_test
def _plot_scv_rs_matrix(freqs, t_inds, scv_rs):

    plot_scv_rs_matrix(freqs, t_inds, scv_rs,
                       save_fig=True, file_path=TEST_PLOTS_PATH,
                       file_name='test_plot_scv_rs_matrix.png')

@plot_test
def test_plot_spectral_hist(tsig_comb):

    freqs, power_bins, spectral_hist = compute_spectral_hist(tsig_comb, fs=FS)
    spectrum_freqs, spectrum = compute_spectrum(tsig_comb, fs=FS)

    plot_spectral_hist(freqs, power_bins, spectral_hist,
                       spectrum=spectrum, spectrum_freqs=spectrum_freqs,
                       save_fig=True, file_path=TEST_PLOTS_PATH,
                       file_name='test_plot_spectral_hist.png')
