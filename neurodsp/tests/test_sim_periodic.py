"""Test periodic simulation functions."""

from neurodsp.sim.periodic import *
from neurodsp.sim.periodic import _make_is_osc, _determine_cycle_properties, _sim_cycles

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
    pass

    sig = sim_jittered_oscillation(N_SECONDS, FS, FREQ)
    assert True

## PRIVATE FUNCTIONS

def test_make_is_osc():
    pass

def test_determine_cycle_properties():
    pass

def test_sim_cycles():
    pass
