"""Test periodic simulation functions."""

from neurodsp.tests.tutils import check_sim_output
from neurodsp.tests.settings import FS, N_SECONDS, FREQ1

from neurodsp.sim.periodic import *
from neurodsp.sim.periodic import _make_is_osc

###################################################################################################
###################################################################################################

def test_sim_oscillation():

    sig = sim_oscillation(N_SECONDS, FS, FREQ1)
    check_sim_output(sig)

def test_sim_bursty_oscillation():

    sig = sim_bursty_oscillation(N_SECONDS, FS, FREQ1)
    check_sim_output(sig)

def test_make_is_osc():

    is_osc = _make_is_osc(10, 0.5, 0.5)
    assert isinstance(is_osc[0], bool)
