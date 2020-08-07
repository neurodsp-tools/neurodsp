"""Test aperiodic simulation functions."""

from neurodsp.tests.settings import FS, N_SECONDS, N_SECONDS_LONG
from neurodsp.tests.tutils import check_sim_output

from neurodsp.sim.aperiodic import *
from neurodsp.sim.aperiodic import _create_powerlaw

import numpy as np
from scipy.stats import skew, kurtosis

###################################################################################################
###################################################################################################

def test_sim_poisson_pop():

    sig = sim_poisson_pop(N_SECONDS, FS)
    check_sim_output(sig)

def test_sim_synaptic_current():

    sig = sim_synaptic_current(N_SECONDS, FS)
    check_sim_output(sig)

def test_sim_random_walk():

    sig = sim_random_walk(N_SECONDS, FS)
    check_sim_output(sig)

def test_sim_powerlaw():

    sig = sim_powerlaw(N_SECONDS, FS)
    check_sim_output(sig)

    # Test with a filter applied
    sig = sim_powerlaw(N_SECONDS, FS, f_range=(2, None))
    check_sim_output(sig)

def test_sim_frac_gaussian_noise():

    # Simulate white noise. Do not normalize.
    sig = sim_frac_gaussian_noise(N_SECONDS_LONG, FS, mean=None, variance=None)

    # Check the accuracy of the mean and standard deviation
    np.allclose(np.mean(sig), 0, atol=0.01)
    np.allclose(np.std(sig), 1, atol=0.01)
    np.allclose(skew(sig), 0, atol=0.01)
    np.allclose(kurtosis(sig), 3, atol=0.01)

def test_sim_frac_brownian_motion():

    # Simulate standard brownian motion. Do not normalize.
    sig = sim_frac_brownian_motion(N_SECONDS_LONG, FS)

    # Check the accuracy of the mean and standard deviation of the increments
    np.allclose(np.mean(np.diff(sig)), 0, atol=0.01)
    np.allclose(np.std(np.diff(sig)), 1, atol=0.01)
    np.allclose(skew(sig), 0, atol=0.01)
    np.allclose(kurtosis(sig), 3, atol=0.01)

def test_create_powerlaw():

    sig = _create_powerlaw(int(N_SECONDS*FS), FS, -2)
    check_sim_output(sig)
