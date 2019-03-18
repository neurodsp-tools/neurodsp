"""Test periodic simulation functions."""

import os

import numpy as np

import neurodsp
from neurodsp.sim.periodic import *

###################################################################################################
###################################################################################################

fs = 1000
freq = 6
rdsym = .5
n_seconds = 10

def test_sim_oscillation():

    np.random.seed(0)
    osc = sim_oscillation(n_seconds, fs, freq, rdsym=rdsym)
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_osc.npy', osc)
    osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_osc.npy')

    assert np.allclose(np.sum(np.abs(osc - osc_true)), 0, atol=10 ** -5)

def test_sim_bursty_oscillation():

    np.random.seed(0)
    osc = sim_bursty_oscillation(n_seconds, fs, freq, rdsym=rdsym)
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_bursty_osc.npy', osc)
    osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_bursty_osc.npy')

    assert np.allclose(np.sum(np.abs(osc - osc_true)), 0, atol=10 ** -5)

def test_sim_jittered_oscillation():

    np.random.seed(0)
    jittered_osc = sim_jittered_oscillation(2, 1000, 20, 0.00, ('gaussian', 0.01))
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_jittered_oscillator.npy', jittered_osc)
    jittered_osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_jittered_oscillator.npy')

    assert np.allclose(np.sum(np.abs(jittered_osc - jittered_osc_true)), 0, atol=10 ** -5)
