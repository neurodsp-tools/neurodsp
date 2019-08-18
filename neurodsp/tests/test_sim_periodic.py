"""Test periodic simulation functions."""

from neurodsp.tests.utils import FS, N_SECONDS, FREQ
from neurodsp.tests.utils import check_sim_output

from neurodsp.sim.periodic import *
from neurodsp.sim.periodic import _make_is_osc

###################################################################################################
###################################################################################################

def test_sim_oscillation():

    sig = sim_oscillation(N_SECONDS, FS, FREQ)
    check_sim_output(sig)

def test_sim_bursty_oscillation():

    sig = sim_bursty_oscillation(N_SECONDS, FS, FREQ)
    check_sim_output(sig)

def test_make_is_osc():
    pass
