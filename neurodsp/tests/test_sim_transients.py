"""Test transient simulation functions."""

import os

import numpy as np

import neurodsp
from neurodsp.sim.transients import *

###################################################################################################
###################################################################################################

fs = 1000

def test_sim_make_synaptic_kernel():

    np.random.seed(0)

    # smoke test that valid parameter configurations do not return negative values
    t_ker, fs, tau_r, tau_d = 1., 1000., 0., 0.02
    assert np.all(make_synaptic_kernel(t_ker, fs, tau_r, tau_d) >= 0.)

    t_ker, fs, tau_r, tau_d = 2., 2000., 0.005, 0.02
    assert np.all(make_synaptic_kernel(t_ker, fs, tau_r, tau_d) >= 0.)

    t_ker, fs, tau_r, tau_d = 1., 1000., 0.02, 0.02
    assert np.all(make_synaptic_kernel(t_ker, fs, tau_r, tau_d) >= 0.)

def test_make_osc_cycle():

    np.random.seed(0)
    gaus_cycle = make_osc_cycle(0.05, 1000., ('gaussian', 0.01))
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/make_osc_cycle.npy', gaus_cycle)
    gaus_cycle_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/make_osc_cycle.npy')

    assert np.allclose(np.sum(np.abs(gaus_cycle - gaus_cycle_true)), 0, atol=10 ** -5)
