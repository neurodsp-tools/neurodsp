"""Test cycle simulation functions."""

from pytest import raises

import numpy as np

from neurodsp.tests.tutils import check_sim_output
from neurodsp.tests.settings import N_SECONDS, FS

from neurodsp.sim.cycles import *

###################################################################################################
###################################################################################################

def test_sim_cycle():

    cycle = sim_cycle(N_SECONDS, FS, 'sine')
    check_sim_output(cycle)

    cycle = sim_cycle(N_SECONDS, FS, 'asine', rdsym=0.75)
    check_sim_output(cycle)

    cycle = sim_cycle(N_SECONDS, FS, 'sawtooth', width=0.5)
    check_sim_output(cycle)

    cycle = sim_cycle(N_SECONDS, FS, 'gaussian', std=2)
    check_sim_output(cycle)

    cycle = sim_cycle(N_SECONDS, FS, 'exp', tau_d=0.2)
    check_sim_output(cycle)

    cycle = sim_cycle(N_SECONDS, FS, '2exp', tau_r=0.2, tau_d=0.2)
    check_sim_output(cycle)

    with raises(ValueError):
        sim_cycle(N_SECONDS, FS, 'not_a_cycle')

def test_sim_sine_cycle():

    cycle = sim_sine_cycle(N_SECONDS, FS)
    check_sim_output(cycle)

def test_sim_asine_cycle():

    cycle = sim_asine_cycle(N_SECONDS, FS, 0.25)
    check_sim_output(cycle)

def test_sim_sawtooth_cycle():

    cycle = sim_sawtooth_cycle(N_SECONDS, FS, 0.5)
    check_sim_output(cycle)

def test_sim_gaussian_cycle():

    cycle = sim_gaussian_cycle(N_SECONDS, FS, 2)
    check_sim_output(cycle)

def test_create_cycle_time():

    times = create_cycle_time(N_SECONDS, FS)
    check_sim_output(times)

def test_phase_shift_cycle():

    cycle = sim_cycle(N_SECONDS, FS, 'sine')

    # Check cycle does not change if not rotated
    cycle_noshift = phase_shift_cycle(cycle, 0.)
    check_sim_output(cycle_noshift)
    assert np.array_equal(cycle, cycle_noshift)

    # Check cycle does change if rotated
    cycle_shifted = phase_shift_cycle(cycle, 0.25)
    check_sim_output(cycle_shifted)
    assert not np.array_equal(cycle, cycle_shifted)
