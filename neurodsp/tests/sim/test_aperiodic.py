"""Test aperiodic simulation functions."""

from neurodsp.tests.settings import FS, N_SECONDS
from neurodsp.tests.tutils import check_sim_output

from neurodsp.sim.aperiodic import *
from neurodsp.sim.aperiodic import _create_powerlaw

import numpy as np

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

def test_sim_fgn():

    # Simulate white noise.
    sig = sim_fgn(N_SECONDS, FS)
    check_sim_output(sig)

    # Check the accuracy of the mean and standard deviation.
    np.allclose(np.mean(sig), 0, atol=0.1)
    np.allclose(np.std(sig), 1, atol=0.1)

def test_sim_fbm():

    # Simulate standard brownian motion.
    sig = sim_fbm(N_SECONDS, FS)
    check_sim_output(sig)

    # Check the accuracy of the mean and standard deviation of the increments.
    np.allclose(np.mean(np.diff(sig)), 0, atol=0.1)
    np.allclose(np.std(np.diff(sig)), 1, atol=0.1)

def test_create_powerlaw():

    sig = _create_powerlaw(int(N_SECONDS*FS), FS, -2)
    check_sim_output(sig)
