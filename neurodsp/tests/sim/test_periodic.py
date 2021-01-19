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

    is_osc = make_is_osc_prob(15, 0.5, 0.5)
    assert is_osc.dtype == 'bool'
    assert sum(is_osc) < len(is_osc)

def test_make_is_osc_durations():

    is_osc = make_is_osc_durations(15, 2, 2)
    assert is_osc.dtype == 'bool'
    assert list(is_osc[1:3]) == [True, True]
    assert list(is_osc[3:5]) == [False, False]
