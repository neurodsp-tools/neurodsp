"""Test aperiodic simulation functions."""

from neurodsp.sim.aperiodic import *

###################################################################################################
###################################################################################################

FS = 100
N_SECONDS = 1

def test_sim_poisson_pop():

    sig = sim_poisson_pop(N_SECONDS, FS)
    assert True

def test_sim_synaptic_current():

    sig = sim_synaptic_current(N_SECONDS, FS)
    assert True

def test_sim_random_walk():

    sig = sim_random_walk(N_SECONDS, FS)
    assert True

def test_powerlaw():

    sig = sim_powerlaw(N_SECONDS, FS)
    assert True
