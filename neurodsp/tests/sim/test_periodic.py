"""Test periodic simulation functions."""

from neurodsp.tests.tutils import check_sim_output
from neurodsp.tests.settings import FS, N_SECONDS, FREQ1

from neurodsp.sim.periodic import *

###################################################################################################
###################################################################################################

def test_sim_oscillation():

    sig = sim_oscillation(N_SECONDS, FS, FREQ1)
    check_sim_output(sig)

def test_sim_bursty_oscillation():

    sig = sim_bursty_oscillation(N_SECONDS, FS, FREQ1)
    check_sim_output(sig)

def test_make_is_osc_prob():

    is_osc = make_is_osc_prob(10, 0.5, 0.5)
    assert is_osc.dtype == 'bool'
