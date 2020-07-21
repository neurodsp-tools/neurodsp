"""Test transient simulation functions."""

import numpy as np

from neurodsp.tests.tutils import check_sim_output
from neurodsp.tests.settings import N_SECONDS, FS

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
