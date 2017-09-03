"""
test_spectral.py
Test functions in the spectral domain analysis module
"""

import numpy as np
import os
import neurodsp
from neurodsp import spectral
from neurodsp.tests import _load_example_data


def test_psd():
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
    freq, Pmean = spectral.psd(x, fs, method='mean', nperseg=fs * 2)
    freq, Pmed = spectral.psd(x, fs, method='median', nperseg=fs * 2)
    freqmf, Pmedfilt = spectral.psd(x, fs, method='medfilt')

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
    fs = 1000.

    # load ground truth scv
    gt_scv = np.load(os.path.dirname(neurodsp.__file__) +
                     '/tests/data/sample_data_' + str(data_idx) + '_scv.npz')

    # compute SCV
    freq, SCV = spectral.scv(x, fs)
    assert np.allclose(np.sum(np.abs(gt_scv['freq'] - freq)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_scv['SCV'] - SCV)), 0, atol=10 ** -5)


def test_scvrs():
    """
    Confirm SCV resampling calculation
    """
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    fs = 1000.

    # load ground truth scv
    gt_scvrs = np.load(os.path.dirname(neurodsp.__file__) +
                       '/tests/data/sample_data_' + str(data_idx) + '_scvrs.npz')

    # compute SCV and compare differences
    np.random.seed(99)
    freqbs, Tbs, SCVrsbs = spectral.scv_rs(x, fs, method='bootstrap', rs_params=(5, 20))
    assert np.allclose(np.sum(np.abs(gt_scvrs['freqbs'] - freqbs)), 0, atol=10 ** -5)
    assert Tbs is None
    assert np.allclose(np.sum(np.abs(gt_scvrs['SCVrsbs'] - SCVrsbs)), 0, atol=10 ** -5)

    freqro, Tro, SCVrsro = spectral.scv_rs(x, fs, method='rolling', rs_params=(4, 2))
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
    fs = 1000.

    # load ground truth scv
    gt_sphist = np.load(os.path.dirname(neurodsp.__file__) +
                        '/tests/data/sample_data_' + str(data_idx) + '_sphist.npz')

    freq, bins, sp_hist = spectral.spectral_hist(x, fs, nbins=10)
    assert np.allclose(np.sum(np.abs(gt_sphist['freq'] - freq)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_sphist['bins'] - bins)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_sphist['sp_hist'] - sp_hist)), 0, atol=10 ** -5)


def test_fitpsd():
    """
    Confirm PSD fitting procedure
    """
    # Load data
    data_idx = 1

    # load test data PSDs for testing
    gt_psd = np.load(os.path.dirname(neurodsp.__file__) +
                     '/tests/data/sample_data_' + str(data_idx) + '_psd.npz')

    psd = gt_psd['PSDmean']
    freq = gt_psd['freq']
    slope_ols, offset_ols = spectral.fit_slope(freq, psd, (30, 100), method='ols')
    slope_rsc, offset_rsc = spectral.fit_slope(freq, psd, (30, 100), method='RANSAC')

    # load ground truth fits
    gt_fitpsd = np.load(os.path.dirname(neurodsp.__file__) +
                        '/tests/data/sample_data_' + str(data_idx) + '_fitpsd.npz')

    assert np.allclose(gt_fitpsd['slope_ols'] - slope_ols, 0, atol=10 ** -5)
    assert np.allclose(gt_fitpsd['slope_rsc'] - slope_rsc, 0, atol=10 ** -5)
    assert np.allclose(gt_fitpsd['offset_ols'] - offset_ols, 0, atol=10 ** -5)
    assert np.allclose(gt_fitpsd['offset_rsc'] - offset_rsc, 0, atol=10 ** -5)
