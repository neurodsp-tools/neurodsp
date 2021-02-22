"""Tests for neurodsp.sim.periodic."""

from neurodsp.tests.tutils import check_sim_output
from neurodsp.tests.settings import FS, N_SECONDS, FREQ1

from neurodsp.sim.periodic import *

###################################################################################################
###################################################################################################

def test_sim_oscillation():

    sig = sim_oscillation(N_SECONDS, FS, FREQ1)
    check_sim_output(sig)

def test_sim_bursty_oscillation():

    # Check default values work
    sig1 = sim_bursty_oscillation(N_SECONDS, FS, FREQ1)
    check_sim_output(sig1)

    # Check probability burst approach
    sig2 = sim_bursty_oscillation(N_SECONDS, FS, FREQ1, \
        burst_def='prob', burst_params={'enter_burst' : 0.3, 'leave_burst' : 0.3})
    check_sim_output(sig2)

    # Check durations burst approach
    sig3 = sim_bursty_oscillation(N_SECONDS, FS, FREQ1, \
        burst_def='durations', burst_params={'n_cycles_burst' : 2, 'n_cycles_off' : 2})
    check_sim_output(sig3)

def test_make_bursts():

    is_osc = np.array([False, False, True, True, False, True, False, True, True, False])
    cycle = np.ones([10])

    sig = make_bursts(N_SECONDS, FS, is_osc, cycle)
    check_sim_output(sig)

def test_make_is_osc_prob():

    is_osc = make_is_osc_prob(15, 0.5, 0.5)
    assert is_osc.dtype == 'bool'
    assert sum(is_osc) < len(is_osc)

def test_make_is_osc_durations():

    is_osc = make_is_osc_durations(15, 2, 2)
    assert is_osc.dtype == 'bool'
    assert list(is_osc[0:2]) == [True, True]
    assert list(is_osc[2:4]) == [False, False]

def test_get_burst_samples():

    is_oscillating = np.array([False, True, True, False])
    burst_samples = get_burst_samples(is_oscillating, FS, 10)

    # First ten samples should be false & next ten samples should by true
    assert sum(burst_samples[0:10]) == 0
    assert sum(burst_samples[10:20]) == 10
