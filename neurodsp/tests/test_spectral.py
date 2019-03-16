"""Test functions in the spectral domain analysis module."""

import os

import numpy as np

import neurodsp
from neurodsp import spectral
from .util import load_example_data

###################################################################################################
###################################################################################################

def test_compute_spectrum():

    # Load data
    data_idx = 1
    sig = load_example_data(data_idx=data_idx)
    fs = 1000.

    # load "ground truth" PSDs
    gt_psd = np.load(os.path.dirname(neurodsp.__file__) +
                     '/tests/data/sample_data_' + str(data_idx) + '_psd.npz')

    # try all 3 different methods
    freq, Pmean = spectral.compute_spectrum(sig, fs, method='mean', nperseg=fs * 2)
    freq, Pmed = spectral.compute_spectrum(sig, fs, method='median', nperseg=fs * 2)
    freqmf, Pmedfilt = spectral.compute_spectrum(sig, fs, method='medfilt')

    assert np.allclose(np.sum(np.abs(gt_psd['PSDmean'] - Pmean)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_psd['PSDmed'] - Pmed)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_psd['PSDmedfilt'] - Pmedfilt)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_psd['freq'] - freq)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_psd['freqmf'] - freqmf)), 0, atol=10 ** -5)

def test_compute_scv():

    # Load data
    data_idx = 1
    sig = load_example_data(data_idx=data_idx)
    fs = 1000.

    # load ground truth scv
    gt_scv = np.load(os.path.dirname(neurodsp.__file__) +
                     '/tests/data/sample_data_' + str(data_idx) + '_scv.npz')

    # compute SCV
    freq, spect_cv = spectral.compute_scv(sig, fs)

    assert np.allclose(np.sum(np.abs(gt_scv['freq'] - freq)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_scv['SCV'] - spect_cv)), 0, atol=10 ** -5)

def test_scv_rs():

    # Load data
    data_idx = 1
    sig = load_example_data(data_idx=data_idx)
    fs = 1000.

    # load ground truth scv
    gt_scv_rs = np.load(os.path.dirname(neurodsp.__file__) +
                        '/tests/data/sample_data_' + str(data_idx) + '_scvrs.npz')

    # compute SCV and compare differences
    np.random.seed(99)
    freq_bs, t_bs, scv_rs_bs = spectral.compute_scv_rs(sig, fs, method='bootstrap', rs_params=(5, 20))

    assert np.allclose(np.sum(np.abs(gt_scv_rs['freqbs'] - freq_bs)), 0, atol=10 ** -5)
    assert t_bs is None
    assert np.allclose(np.sum(np.abs(gt_scv_rs['SCVrsbs'] - scv_rs_bs)), 0, atol=10 ** -5)

    freq_ro, t_ro, scv_rs_ro = spectral.compute_scv_rs(sig, fs, method='rolling', rs_params=(4, 2))

    assert np.allclose(np.sum(np.abs(gt_scv_rs['freqro'] - freq_ro)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_scv_rs['Tro'] - t_ro)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_scv_rs['SCVrsro'] - scv_rs_ro)), 0, atol=10 ** -5)

def test_spectralhist():

    # Load data
    data_idx = 1
    sig = load_example_data(data_idx=data_idx)
    fs = 1000.

    # load ground truth scv
    gt_sphist = np.load(os.path.dirname(neurodsp.__file__) +
                        '/tests/data/sample_data_' + str(data_idx) + '_sphist.npz')

    freq, bins, sp_hist = spectral.spectral_hist(sig, fs, nbins=10)

    assert np.allclose(np.sum(np.abs(gt_sphist['freq'] - freq)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_sphist['bins'] - bins)), 0, atol=10 ** -5)
    assert np.allclose(np.sum(np.abs(gt_sphist['sp_hist'] - sp_hist)), 0, atol=10 ** -5)

def test_rotatepsd():

    rot_exp = -2
    spectrum = np.ones(500)
    f_axis = np.arange(0, 500.)
    spectrum_rot = spectral.rotate_powerlaw(f_axis, spectrum, rot_exp)

    # load test data PSDs for testing
    spectrum_test = np.load(os.path.dirname(neurodsp.__file__) +
                            '/tests/data/sim_rotatepsd.npy')

    assert np.allclose(spectrum_rot - spectrum_test, 0, atol=10 ** -5)

def test_morlet_transform():

    data_idx = 1
    sig = load_example_data(data_idx=data_idx)
    fs = 1000.

    morlet_freqs = np.logspace(0, 7, 15, base=2)
    mwt = spectral.morlet_transform(sig, morlet_freqs, fs)

    #np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/mwt.npy', mwt)
    # load "ground truth" MWT
    gt_mwt = np.load(os.path.dirname(neurodsp.__file__) +
                     '/tests/data/mwt.npy')

    assert np.allclose(np.sum(np.abs(gt_mwt - mwt)), 0, atol=10 ** -5)
