""".  """

from neurodsp.spectral.scv import *

###################################################################################################
###################################################################################################

def test_compute_scv(tsig):

    freqs, spect_cv = compute_scv(tsig, fs=500)
    assert True

def test_compute_scv_rs(tsig):

    freqs, t_inds, spect_cv = compute_scv_rs(tsig, fs=500, method='bootstrap')
    freqs, t_inds, spect_cv = compute_scv_rs(tsig, fs=500, method='rolling')
    assert True

def test_compute_spectral_hist(tsig):

    freqs, bins, spect_hist = compute_spectral_hist(tsig, fs=500)
    assert True
