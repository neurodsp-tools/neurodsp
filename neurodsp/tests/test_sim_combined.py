"""Test combined simulation functions."""

import os

import numpy as np

import neurodsp
from neurodsp.sim.combined import *

###################################################################################################
###################################################################################################

fs = 1000
freq = 6
n_seconds = 10
exponent = -2
f_range_filter = (2, None)
filter_order = 1501
ratio_osc_var = 1

def test_sim_noisy_oscillation():

    np.random.seed(0)
    osc = sim_noisy_oscillation(n_seconds, fs, freq, 'filtered_powerlaw',
                               {'exponent': exponent,
                                'f_range': f_range_filter,
                                'filter_order': filter_order},
                               ratio_osc_var=ratio_osc_var)
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_noisy_osc.npy', osc)
    osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_noisy_osc.npy')

    assert np.allclose(np.sum(np.abs(osc - osc_true)), 0, atol=10 ** -5)


def test_sim_noisy_bursty_oscillation():

    np.random.seed(0)
    osc = sim_noisy_bursty_oscillation(n_seconds, fs, freq, 'filtered_powerlaw',
                                      {'exponent': exponent,
                                       'f_range': f_range_filter,
                                       'filter_order': filter_order},
                                      rdsym=.5, ratio_osc_var=1, prob_enter_burst=.2, prob_leave_burst=.2,
                                      cycle_features=None, return_components=False, return_cycle_df=False)

    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_noisy_bursty_osc.npy', osc)
    osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_noisy_bursty_osc.npy')

    assert np.allclose(np.sum(np.abs(osc - osc_true)), 0, atol=10 ** -5)
