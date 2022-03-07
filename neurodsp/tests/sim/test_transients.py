"""Tests for neurodsp.sim.transients."""

import numpy as np

from neurodsp.tests.tutils import check_sim_output
from neurodsp.tests.settings import N_SECONDS, N_SECONDS_ODD, FS, FS_ODD
from neurodsp.sim.transients import *

###################################################################################################
###################################################################################################

def test_sim_synaptic_kernel():

    # Check that valid parameter configurations do not return negative values
    for params in [[0., 0.02], [0.005, 0.02], [0.02, 0.02]]:
        tau_r, tau_d = params
        kernel = sim_synaptic_kernel(N_SECONDS, FS, tau_r, tau_d)
        check_sim_output(kernel)
        assert np.all(kernel >= 0.)

def test_sim_action_potential():

    stds = (.25, .2)
    alphas = (8, .2)
    centers = (.25, .5)
    heights = (15, 2.5)

    cycle = sim_action_potential(N_SECONDS, FS, centers, stds, alphas, heights)
    check_sim_output(cycle, n_seconds=N_SECONDS)

    cycle = sim_action_potential(N_SECONDS_ODD, FS, centers, stds, alphas, heights)
    check_sim_output(cycle, n_seconds=N_SECONDS_ODD)

    cycle = sim_action_potential(N_SECONDS, FS_ODD, centers, stds, alphas, heights)
    check_sim_output(cycle, n_seconds=N_SECONDS, fs=FS_ODD)

    cycle = sim_action_potential(N_SECONDS, FS, centers, stds, alphas, heights)
    check_sim_output(cycle, n_seconds=N_SECONDS)


def test_sim_damped_erp():

    erp = sim_damped_erp(N_SECONDS, FS, amp=1, freq=5, decay=0.05)
    check_sim_output(erp, n_seconds=N_SECONDS)

    erp = sim_damped_erp(N_SECONDS, FS, amp=1, freq=5, decay=0.05, time_start=1)
    assert not np.any(erp[:int(1 * FS)])
