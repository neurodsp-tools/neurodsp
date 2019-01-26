"""Test simulation functions.

Code to make true data files:

import numpy as np
import os
import neurodsp
from neurodsp.sim import *

n_seconds = 10
fs = 1000
freq = 6
rdsym = .5
f_range_filter = (2, None)
filter_order = 1501
exponent = 2
ratio_osc_var = 1

np.random.seed(0)
noise = sim_filtered_noise(n_seconds, fs, f_range_filter, filter_order, exponent=exponent)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/noise_filt.npy', noise)

np.random.seed(0)
osc = sim_oscillator(n_seconds, fs, freq, rdsym=rdsym)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_osc.npy', osc)

np.random.seed(0)
osc = sim_noisy_oscillator(n_seconds, fs, freq, exponent=exponent,
                           rdsym=rdsym, f_range_filter=f_range_filter,
                           ratio_osc_var=ratio_osc_var, filter_order=filter_order)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_noisy_osc.npy', osc)

np.random.seed(0)
osc = sim_bursty_oscillator(n_seconds, fs, freq, rdsym=rdsym)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_bursty_osc.npy', osc)

np.random.seed(0)
osc = sim_noisy_bursty_oscillator(n_seconds, fs, freq, rdsym=rdsym, exponent=exponent,
                                  f_range_filter=f_range_filter, ratio_osc_var=ratio_osc_var,
                                  prob_enter_burst=.2, prob_leave_burst=.2,
                                  cycle_features=None, return_components=False,
                                  return_cycle_df=False)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_noisy_bursty_osc.npy', osc)

np.random.seed(0)
syn_noise = sim_synaptic_noise(2, 1000, 1000, 2, 1., 0.002, 2)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_synaptic_noise.npy', syn_noise)

np.random.seed(0)
ou_noise = sim_ou_process(2, 1000, 1., 0., 5.)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_ou_process.npy', ou_noise)

np.random.seed(0)
jittered_osc = sim_jittered_oscillator(2, 1000, 20, 0.00, ('gaussian',0.01))
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_jittered_oscillator.npy', jittered_osc)

np.random.seed(0)
powerlaw = sim_variable_powerlaw(60, 1000, -2.25)
np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_variable_powerlaw.npy', powerlaw)

"""
import os

import numpy as np

import neurodsp
from neurodsp.sim import *

###################################################################################################
###################################################################################################

n_seconds = 10
fs = 1000
freq = 6
rdsym = .5
f_range_filter = (2, None)
filter_order = 1501
exponent = -2
ratio_osc_var = 1


def test_sim_filtered_noise():

    np.random.seed(0)
    noise = sim_filtered_noise(n_seconds, fs, exponent, f_range_filter, filter_order)
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/noise_filt.npy', noise)
    noise_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/noise_filt.npy')

    assert np.allclose(np.sum(np.abs(noise - noise_true)), 0, atol=10 ** -5)


def test_sim_oscillator():

    np.random.seed(0)
    osc = sim_oscillator(n_seconds, fs, freq, rdsym=rdsym)
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_osc.npy', osc)
    osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_osc.npy')

    assert np.allclose(np.sum(np.abs(osc - osc_true)), 0, atol=10 ** -5)


def test_sim_noisy_oscillator():

    np.random.seed(0)
    osc = sim_noisy_oscillator(n_seconds, fs, freq, 'filtered_powerlaw',
                               {'exponent': exponent,
                                'f_range': f_range_filter,
                                'filter_order': filter_order},
                               ratio_osc_var=ratio_osc_var)

    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_noisy_osc.npy', osc)
    osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_noisy_osc.npy')

    assert np.allclose(np.sum(np.abs(osc - osc_true)), 0, atol=10 ** -5)


def test_sim_bursty_oscillator():

    np.random.seed(0)
    osc = sim_bursty_oscillator(n_seconds, fs, freq, rdsym=rdsym)
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_bursty_osc.npy', osc)
    osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_bursty_osc.npy')

    assert np.allclose(np.sum(np.abs(osc - osc_true)), 0, atol=10 ** -5)


def test_sim_noisy_bursty_oscillator():

    np.random.seed(0)
    osc = sim_noisy_bursty_oscillator(n_seconds, fs, freq, 'filtered_powerlaw',
                                      {'exponent': exponent,
                                       'f_range': f_range_filter,
                                       'filter_order': filter_order},
                                      rdsym=.5, ratio_osc_var=1, prob_enter_burst=.2, prob_leave_burst=.2,
                                      cycle_features=None, return_components=False, return_cycle_df=False)

    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_noisy_bursty_osc.npy', osc)
    osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_noisy_bursty_osc.npy')

    assert np.allclose(np.sum(np.abs(osc - osc_true)), 0, atol=10 ** -5)


def test_sim_poisson_pop():

    np.random.seed(0)
    poisson_noise = sim_poisson_pop(2., 1000., 100., 4.)
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_poisson_pop.npy', poisson_noise)
    poisson_noise_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_poisson_pop.npy')

    assert np.allclose(np.sum(np.abs(poisson_noise - poisson_noise_true)), 0, atol=10 ** -5)


def test_sim_make_synaptic_kernel():

    np.random.seed(0)

    # smoke test that valid parameter configurations do not return negative values
    t_ker, fs, tau_r, tau_d = 1., 1000., 0., 0.02
    assert np.all(make_synaptic_kernel(t_ker, fs, tau_r, tau_d) >= 0.)

    t_ker, fs, tau_r, tau_d = 2., 2000., 0.005, 0.02
    assert np.all(make_synaptic_kernel(t_ker, fs, tau_r, tau_d) >= 0.)

    t_ker, fs, tau_r, tau_d = 1., 1000., 0.02, 0.02
    assert np.all(make_synaptic_kernel(t_ker, fs, tau_r, tau_d) >= 0.)


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


def test_sim_jittered_oscillator():

    np.random.seed(0)
    jittered_osc = sim_jittered_oscillator(2, 1000, 20, 0.00, ('gaussian', 0.01))
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_jittered_oscillator.npy', jittered_osc)
    jittered_osc_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_jittered_oscillator.npy')

    assert np.allclose(np.sum(np.abs(jittered_osc - jittered_osc_true)), 0, atol=10 ** -5)


def test_make_osc_cycle():

    np.random.seed(0)
    gaus_cycle = make_osc_cycle(0.05, 1000., ('gaussian', 0.01))
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/make_osc_cycle.npy', gaus_cycle)
    gaus_cycle_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/make_osc_cycle.npy')

    assert np.allclose(np.sum(np.abs(gaus_cycle - gaus_cycle_true)), 0, atol=10 ** -5)


def test_variable_powerlaw():

    np.random.seed(0)
    powerlaw = sim_variable_powerlaw(60, 1000, -2.25)
    # np.save(os.path.dirname(neurodsp.__file__) + '/tests/data/sim_variable_powerlaw.npy', powerlaw)
    powerlaw_true = np.load(os.path.dirname(
        neurodsp.__file__) + '/tests/data/sim_variable_powerlaw.npy')

    assert np.allclose(np.sum(np.abs(powerlaw - powerlaw_true)), 0, atol=10 ** -5)
