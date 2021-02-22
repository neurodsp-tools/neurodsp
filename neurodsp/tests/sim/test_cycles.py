"""Test cycle simulation functions."""

from pytest import raises

import numpy as np

from neurodsp.tests.tutils import check_sim_output
from neurodsp.tests.settings import N_SECONDS, FS, N_SECONDS_ODD, FS_ODD

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

    cycle = sim_cycle(1/7, FS, 'gaussian', std=2)
    check_sim_output(cycle, n_seconds=1/7)

def test_sim_normalized_cycle():

    check_sim_output(sim_normalized_cycle(N_SECONDS, FS, 'sine'))
    check_sim_output(sim_normalized_cycle(N_SECONDS_ODD, FS, 'sine'), n_seconds=N_SECONDS_ODD)
    check_sim_output(sim_normalized_cycle(N_SECONDS, FS_ODD, 'sine'), fs=FS_ODD)

def test_sim_sine_cycle():

    check_sim_output(sim_sine_cycle(N_SECONDS, FS))
    check_sim_output(sim_sine_cycle(N_SECONDS_ODD, FS), n_seconds=N_SECONDS_ODD)
    check_sim_output(sim_sine_cycle(N_SECONDS, FS_ODD), fs=FS_ODD)

def test_sim_asine_cycle():

    check_sim_output(sim_asine_cycle(N_SECONDS, FS, 0.25))
    check_sim_output(sim_asine_cycle(N_SECONDS_ODD, FS, 0.25), n_seconds=N_SECONDS_ODD)
    check_sim_output(sim_asine_cycle(N_SECONDS, FS_ODD, 0.25), fs=FS_ODD)

def test_sim_sawtooth_cycle():

    check_sim_output(sim_sawtooth_cycle(N_SECONDS, FS, 0.75))
    check_sim_output(sim_sawtooth_cycle(N_SECONDS_ODD, FS, 0.75), n_seconds=N_SECONDS_ODD)
    check_sim_output(sim_sawtooth_cycle(N_SECONDS, FS_ODD, 0.75), fs=FS_ODD)

def test_sim_gaussian_cycle():

    check_sim_output(sim_gaussian_cycle(N_SECONDS, FS, 2))
    check_sim_output(sim_gaussian_cycle(N_SECONDS_ODD, FS, 2), n_seconds=N_SECONDS_ODD)
    check_sim_output(sim_gaussian_cycle(N_SECONDS, FS_ODD, 2), fs=FS_ODD)

def test_create_cycle_time():

    check_sim_output(create_cycle_time(N_SECONDS, FS))
    check_sim_output(create_cycle_time(N_SECONDS_ODD, FS), n_seconds=N_SECONDS_ODD)
    check_sim_output(create_cycle_time(N_SECONDS, FS_ODD), fs=FS_ODD)

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

    # Check min-to-min sim
    cycle_shifted = phase_shift_cycle(cycle, 'min')
    check_sim_output(cycle_shifted)
    assert np.argmin(cycle_shifted) == 0

    # Check max-to-mix sim
    cycle_shifted = phase_shift_cycle(cycle, 'max')
    check_sim_output(cycle_shifted)
    assert np.argmax(cycle_shifted) == 0
