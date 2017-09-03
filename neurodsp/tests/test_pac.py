"""
test_pac.py
Test functions in the phase-amplitude coupling module
"""

import numpy as np
import os
import neurodsp
from neurodsp.tests import _load_example_data


def test_pac_consistent():
    """
    Confirm consistency in estimation of pac
    with computations in previous versions
    """

    # Compute pacs
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    Fs = 1000

    # Compute pacs
    all_pac_methods = ['ozkurt', 'plv', 'glm', 'tort', 'canolty']
    f_range_lo = (13, 30)
    f_range_hi = (50, 200)
    N_seconds_lo = .25
    N_seconds_hi = .2

    np.random.seed(0)
    pacs = np.zeros(len(all_pac_methods))
    for i, m in enumerate(all_pac_methods):
        pacs[i] = neurodsp.compute_pac(x, x, Fs, f_range_lo, f_range_hi,
                                       N_seconds_lo=N_seconds_lo, N_seconds_hi=N_seconds_hi,
                                       pac_method=m)

    # Load ground truth pacs
    pacs_true = np.load(os.path.dirname(neurodsp.__file__) +
                        '/tests/data/sample_data_' + str(data_idx) + '_pacs.npy')

    # Compute difference between current and past signals
    assert np.allclose(np.sum(np.abs(pacs - pacs_true)), 0, atol=10 ** -5)


def test_comodulogram_consistent():
    """
    Confirm consistency in estimation of pac comodulogram
    with computations in previous versions
    """

    # Compute pacs
    # Load data
    data_idx = 1
    x = _load_example_data(data_idx=data_idx)
    Fs = 1000

    # Parameters for comodulogram function
    f_pha_bin_edges = np.arange(2, 42, 2)
    f_amp_bin_edges = np.arange(20, 200, 4)
    N_cycles_pha = 5
    N_cycles_amp = 11

    # Compute comodulogram
    comod = neurodsp.compute_pac_comodulogram(x, x, Fs,
                                              f_pha_bin_edges, f_amp_bin_edges,
                                              N_cycles_pha=N_cycles_pha,
                                              N_cycles_amp=N_cycles_amp,
                                              pac_method='ozkurt')

    # Load ground truth comodulogram
    comod_true = np.load(os.path.dirname(neurodsp.__file__) +
                         '/tests/data/sample_data_' + str(data_idx) + '_comod.npy')

    # Compute difference between current and past signals
    assert np.allclose(np.sum(np.abs(comod - comod_true)), 0, atol=10 ** -5)
