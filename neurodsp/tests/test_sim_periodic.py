"""Test periodic simulation functions."""

from neurodsp.sim.periodic import *

###################################################################################################
###################################################################################################

FS = 100
N_SECONDS = 1
FREQ = 10

def test_sim_oscillation():

    sig = sim_oscillation(N_SECONDS, FS, FREQ)
    assert True

def test_sim_bursty_oscillation():

    sig = sim_bursty_oscillation(N_SECONDS, FS, FREQ)
    assert True

def test_sim_jittered_oscillation():

    sig = sim_jittered_oscillation(N_SECONDS, FS, FREQ)
    assert True
