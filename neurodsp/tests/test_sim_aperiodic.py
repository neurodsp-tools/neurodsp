"""Test aperiodic simulation functions."""

import os

import numpy as np

import neurodsp
from neurodsp.sim.aperiodic import *

###################################################################################################
###################################################################################################

fs = 1000
n_seconds = 10
exponent = -2
f_range_filter = (2, None)
filter_order = 1501

def test_sim_filtered_noise():

    noise = sim_filtered_noise(n_seconds, fs, exponent, f_range_filter)
    assert np.all(noise)

    # Note: old consistency test turned off after moving to using filter_signal which
    #   is slightly different than the filter defined directly in `sim_filtered_noise`

    #np.random.seed(0)
    #noise = sim_filtered_noise(n_seconds, fs, exponent, f_range_filter, filter_order)
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/noise_filt.npy', noise)
    #noise_true = np.load(os.path.dirname(
    #    neurodsp.__file__) + '/tests/data/noise_filt.npy')

    #assert np.allclose(np.sum(np.abs(noise - noise_true)), 0, atol=10 ** -5)

def test_sim_poisson_pop():

    np.random.seed(0)
    poisson_noise = sim_poisson_pop(2., 1000., 100., 4.)
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_poisson_pop.npy', poisson_noise)
    poisson_noise_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_poisson_pop.npy')

    assert np.allclose(np.sum(np.abs(poisson_noise - poisson_noise_true)), 0, atol=10 ** -5)

def test_sim_synaptic_noise():

    np.random.seed(0)
    syn_noise = sim_synaptic_noise(2, 1000, 1000, 2, 0.002, 2, 1.)
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_synaptic_noise.npy', syn_noise)
    syn_noise_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_synaptic_noise.npy')

    assert np.allclose(np.sum(np.abs(syn_noise - syn_noise_true)), 0, atol=10 ** -5)

def test_sim_ou_process():

    np.random.seed(0)
    ou_noise = sim_ou_process(2, 1000, 1., 0., 5.)
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_ou_process.npy', ou_noise)
    ou_noise_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_OU_process.npy')

    assert np.allclose(np.sum(np.abs(ou_noise - ou_noise_true)), 0, atol=10 ** -5)

def test_variable_powerlaw():
    pass

    # Note: turned off consistency test after updating rotate_spectrum

    # np.random.seed(0)
    # powerlaw = sim_variable_powerlaw(60, 1000, -2.25)
    # # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_variable_powerlaw.npy', powerlaw)
    # powerlaw_true = np.load(os.path.dirname(
    #     neurodsp.__file__) + '/tests/data/sim_variable_powerlaw.npy')

    # assert np.allclose(np.sum(np.abs(powerlaw - powerlaw_true)), 0, atol=10 ** -5)
