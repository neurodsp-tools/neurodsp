"""
test_spectral.py
Test functions in the spectral domain analysis module
"""

import numpy as np
import os
import neurodsp
from neurodsp import spectral
from neurodsp.tests import _load_example_data


def test_compute_spectrum():
    """
    Confirm consistency in PSD computation
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    fs = 1000.

    # load "ground truth" PSDs
    gt_psd = np.load(os.path.dirname(neurodsp.__file__) +
                     '/tests/data/sample_data_' + str(data_idx) + '_psd.npz')

    # try all 3 different methods
    freq, Pmean = spectral.compute_spectrum(x, fs, method='mean', nperseg=fs * 2)
    freq, Pmed = spectral.compute_spectrum(x, fs, method='median', nperseg=fs * 2)
    freqmf, Pmedfilt = spectral.compute_spectrum(x, fs, method='medfilt')

    # compute the difference
    assert np.allclose(np.sum(np.abs(gt_psd['PSDmean'] - Pmean)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_psd['PSDmed'] - Pmed)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_psd['PSDmedfilt'] - Pmedfilt)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_psd['freq'] - freq)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_psd['freqmf'] - freqmf)), 0, atol=10 ** -5)


def test_compute_scv():
    """
    Confirm SCV calculation
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    fs = 1000.

    # load ground truth scv
    gt_scv = np.load(os.path.dirname(neurodsp.__file__) +
                     '/tests/data/sample_data_' + str(data_idx) + '_scv.npz')

    # compute SCV
    freq, spect_cv = spectral.compute_scv(x, fs)
    assert np.allclose(np.sum(np.abs(gt_scv['freq'] - freq)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_scv['SCV'] - spect_cv)), 0, atol=10 ** -5)


def test_scv_rs():
    """
    Confirm SCV resampling calculation
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    fs = 1000.

    # load ground truth scv
    gt_scv_rs = np.load(os.path.dirname(neurodsp.__file__) +
                       '/tests/data/sample_data_' + str(data_idx) + '_scvrs.npz')

    # compute SCV and compare differences
    np.random.seed(99)
    freqbs, Tbs, scv_rsbs = spectral.compute_scv_rs(x, fs, method='bootstrap', rs_params=(5, 20))
    assert np.allclose(np.sum(np.abs(gt_scv_rs['freqbs'] - freqbs)), 0, atol=10 ** -5)
    assert Tbs is None
    assert np.allclose(np.sum(np.abs(gt_scv_rs['SCVrsbs'] - scv_rsbs)), 0, atol=10 ** -5)

    freqro, Tro, scv_rsro = spectral.compute_scv_rs(x, fs, method='rolling', rs_params=(4, 2))
    assert np.allclose(np.sum(np.abs(gt_scv_rs['freqro'] - freqro)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_scv_rs['Tro'] - Tro)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_scv_rs['SCVrsro'] - scv_rsro)), 0, atol=10 ** -5)


def test_spectralhist():
    """
    Confirm SCV resampling calculation
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    fs = 1000.

    # load ground truth scv
    gt_sphist = np.load(os.path.dirname(neurodsp.__file__) +
                        '/tests/data/sample_data_' + str(data_idx) + '_sphist.npz')

    freq, bins, sp_hist = spectral.spectral_hist(x, fs, nbins=10)
    assert np.allclose(np.sum(np.abs(gt_sphist['freq'] - freq)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_sphist['bins'] - bins)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_sphist['sp_hist'] - sp_hist)), 0, atol=10 ** -5)


def test_rotatepsd():
    """
    Confirm PSD rotation procedure.
    """
    rot_exp = -2
    P = np.ones(500)
    f_axis = np.arange(0,500.)
    P_rot = neurodsp.spectral.rotate_powerlaw(f_axis,P,rot_exp)

    # load test data PSDs for testing
    P_test = np.load(os.path.dirname(neurodsp.__file__) +
                     '/tests/data/sim_rotatepsd.npy')


    assert np.allclose(P_rot - P_test, 0, atol=10 ** -5)
