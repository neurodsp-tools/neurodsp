"""Tests for neurodsp.sim.cycle."""

import pytest
from pytest import raises

import numpy as np

from neurodsp.tests.tutils import check_sim_output
from neurodsp.tests.settings import N_SECONDS_CYCLE, N_SECONDS_ODD, FS, FS_ODD

from neurodsp.sim.cycles import *

###################################################################################################
###################################################################################################

def test_sim_cycle():

    cycle = sim_cycle(N_SECONDS_CYCLE, FS, 'sine')
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_cycle(N_SECONDS_CYCLE, FS, 'asine', rdsym=0.75)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_cycle(N_SECONDS_CYCLE, FS, 'sawtooth', width=0.5)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_cycle(N_SECONDS_CYCLE, FS, 'gaussian', std=2)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_cycle(N_SECONDS_CYCLE, FS, 'exp', tau_d=0.2)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_cycle(N_SECONDS_CYCLE, FS, '2exp', tau_r=0.2, tau_d=0.2)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    with raises(ValueError):
        sim_cycle(N_SECONDS_CYCLE, FS, 'not_a_cycle')

    cycle = sim_cycle(N_SECONDS_ODD, FS, 'gaussian', std=2)
    check_sim_output(cycle, n_seconds=N_SECONDS_ODD)

def test_sim_normalized_cycle():

    cycle = sim_normalized_cycle(N_SECONDS_CYCLE, FS, 'sine')
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_normalized_cycle(N_SECONDS_ODD, FS, 'sine')
    check_sim_output(cycle, n_seconds=N_SECONDS_ODD)

    cycle = sim_normalized_cycle(N_SECONDS_CYCLE, FS_ODD, 'sine')
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE, fs=FS_ODD)

def test_sim_sine_cycle():

    cycle = sim_sine_cycle(N_SECONDS_CYCLE, FS)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_sine_cycle(N_SECONDS_ODD, FS)
    check_sim_output(cycle, n_seconds=N_SECONDS_ODD)

    cycle = sim_sine_cycle(N_SECONDS_CYCLE, FS_ODD)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE, fs=FS_ODD)

@pytest.mark.parametrize('side', ['both', 'peak', 'trough'])
def test_sim_asine_cycle(side):

    cycle = sim_asine_cycle(N_SECONDS_CYCLE, FS, 0.25, side=side)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_asine_cycle(N_SECONDS_ODD, FS, 0.25, side=side)
    check_sim_output(cycle, n_seconds=N_SECONDS_ODD)

    cycle = sim_asine_cycle(N_SECONDS_CYCLE, FS_ODD, 0.25, side=side)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE, fs=FS_ODD)

def test_sim_sawtooth_cycle():

    cycle = sim_sawtooth_cycle(N_SECONDS_CYCLE, FS, 0.75)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_sawtooth_cycle(N_SECONDS_ODD, FS, 0.75)
    check_sim_output(cycle, n_seconds=N_SECONDS_ODD)

    cycle = sim_sawtooth_cycle(N_SECONDS_CYCLE, FS_ODD, 0.75)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE, fs=FS_ODD)

def test_sim_gaussian_cycle():

    cycle = sim_gaussian_cycle(N_SECONDS_CYCLE, FS, 2)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_gaussian_cycle(N_SECONDS_ODD, FS, 2)
    check_sim_output(cycle, n_seconds=N_SECONDS_ODD)

    cycle = sim_gaussian_cycle(N_SECONDS_CYCLE, FS_ODD, 2)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE, fs=FS_ODD)

def test_sim_skewed_gaussian_cycle():

    cycle = sim_skewed_gaussian_cycle(N_SECONDS_CYCLE, FS, 0.5, 0.25, 2)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_skewed_gaussian_cycle(N_SECONDS_ODD, FS, 0.5, 0.25, 2)
    check_sim_output(cycle, n_seconds=N_SECONDS_ODD)

    cycle = sim_skewed_gaussian_cycle(N_SECONDS_CYCLE, FS_ODD, 0.5, 0.25, 2)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE, fs=FS_ODD)

def test_sim_gaussian_asym_cycle():

    cycle = sim_exp_cos_cycle(N_SECONDS_CYCLE, FS, 0.5)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_exp_cos_cycle(N_SECONDS_ODD, FS, 0.5)
    check_sim_output(cycle, n_seconds=N_SECONDS_ODD)

    cycle = sim_exp_cos_cycle(N_SECONDS_CYCLE, FS_ODD, 0.5)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE, fs=FS_ODD)

def test_sim_harmonic_asym_cycle():

    cycle = sim_asym_harmonic_cycle(N_SECONDS_CYCLE, FS, phi=3*np.pi/2, n_harmonics=2)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE)

    cycle = sim_asym_harmonic_cycle(N_SECONDS_ODD, FS, phi=3*np.pi/2, n_harmonics=2)
    check_sim_output(cycle, n_seconds=N_SECONDS_ODD)

    cycle = sim_asym_harmonic_cycle(N_SECONDS_CYCLE, FS_ODD, phi=3*np.pi/2, n_harmonics=2)
    check_sim_output(cycle, n_seconds=N_SECONDS_CYCLE, fs=FS_ODD)

def test_create_cycle_time():

    times = create_cycle_time(N_SECONDS_CYCLE, FS)
    check_sim_output(times, n_seconds=N_SECONDS_CYCLE)

    times = create_cycle_time(N_SECONDS_ODD, FS)
    check_sim_output(times, n_seconds=N_SECONDS_ODD)

    times = create_cycle_time(N_SECONDS_CYCLE, FS_ODD)
    check_sim_output(times, n_seconds=N_SECONDS_CYCLE, fs=FS_ODD)

def test_phase_shift_cycle():

    cycle = sim_cycle(N_SECONDS_CYCLE, FS, 'sine')

    # Check cycle does not change if not rotated
    cycle_noshift = phase_shift_cycle(cycle, 0.)
    check_sim_output(cycle_noshift, n_seconds=N_SECONDS_CYCLE)
    assert np.array_equal(cycle, cycle_noshift)

    # Check cycle does change if rotated
    cycle_shifted = phase_shift_cycle(cycle, 0.25)
    check_sim_output(cycle_shifted, n_seconds=N_SECONDS_CYCLE)
    assert not np.array_equal(cycle, cycle_shifted)

    # Check min-to-min sim
    cycle_shifted = phase_shift_cycle(cycle, 'min')
    check_sim_output(cycle_shifted, n_seconds=N_SECONDS_CYCLE)
    assert np.argmin(cycle_shifted) == 0

    # Check max-to-max sim
    cycle_shifted = phase_shift_cycle(cycle, 'max')
    check_sim_output(cycle_shifted, n_seconds=N_SECONDS_CYCLE)
    assert np.argmax(cycle_shifted) == 0
