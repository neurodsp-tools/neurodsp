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
    s_rate = 1000.

    # load "ground truth" PSDs
    gt_psd = np.load(os.path.dirname(neurodsp.__file__) +
                     '/tests/data/sample_data_' + str(data_idx) + '_psd.npz')

    # try all 3 different methods
    freq, Pmean = spectral.compute_spectrum(x, s_rate, method='mean', nperseg=s_rate * 2)
    freq, Pmed = spectral.compute_spectrum(x, s_rate, method='median', nperseg=s_rate * 2)
    freqmf, Pmedfilt = spectral.compute_spectrum(x, s_rate, method='medfilt')

    # compute the difference
    assert np.allclose(np.sum(np.abs(gt_psd['PSDmean'] - Pmean)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_psd['PSDmed'] - Pmed)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_psd['PSDmedfilt'] - Pmedfilt)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_psd['freq'] - freq)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_psd['freqmf'] - freqmf)), 0, atol=10 ** -5)


def test_scv():
    """
    Confirm SCV calculation
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    s_rate = 1000.

    # load ground truth scv
    gt_scv = np.load(os.path.dirname(neurodsp.__file__) +
                     '/tests/data/sample_data_' + str(data_idx) + '_scv.npz')

    # compute SCV
    freq, SCV = spectral.scv(x, s_rate)
    assert np.allclose(np.sum(np.abs(gt_scv['freq'] - freq)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_scv['SCV'] - SCV)), 0, atol=10 ** -5)


def test_scvrs():
    """
    Confirm SCV resampling calculation
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    s_rate = 1000.

    # load ground truth scv
    gt_scvrs = np.load(os.path.dirname(neurodsp.__file__) +
                       '/tests/data/sample_data_' + str(data_idx) + '_scvrs.npz')

    # compute SCV and compare differences
    np.random.seed(99)
    freqbs, Tbs, SCVrsbs = spectral.scv_rs(x, s_rate, method='bootstrap', rs_params=(5, 20))
    assert np.allclose(np.sum(np.abs(gt_scvrs['freqbs'] - freqbs)), 0, atol=10 ** -5)
    assert Tbs is None
    assert np.allclose(np.sum(np.abs(gt_scvrs['SCVrsbs'] - SCVrsbs)), 0, atol=10 ** -5)

    freqro, Tro, SCVrsro = spectral.scv_rs(x, s_rate, method='rolling', rs_params=(4, 2))
    assert np.allclose(np.sum(np.abs(gt_scvrs['freqro'] - freqro)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_scvrs['Tro'] - Tro)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_scvrs['SCVrsro'] - SCVrsro)), 0, atol=10 ** -5)


def test_spectralhist():
    """
    Confirm SCV resampling calculation
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    s_rate = 1000.

    # load ground truth scv
    gt_sphist = np.load(os.path.dirname(neurodsp.__file__) +
                        '/tests/data/sample_data_' + str(data_idx) + '_sphist.npz')

    freq, bins, sp_hist = spectral.spectral_hist(x, s_rate, nbins=10)
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
