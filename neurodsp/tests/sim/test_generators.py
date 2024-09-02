"""Tests for neurodsp.sim.generators."""

import numpy as np

from neurodsp.sim.aperiodic import sim_powerlaw

from neurodsp.sim.generators import *

###################################################################################################
###################################################################################################

def test_sig_yielder():

    params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}
    yielder = sig_yielder(sim_powerlaw, params, 2)

    for ind, sig in enumerate(yielder):
        assert isinstance(sig, np.ndarray)
    assert ind == 1

def test_sig_sampler():

    params = [{'n_seconds' : 2, 'fs' : 250, 'exponent' : -2},
              {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}]
    sampler = sig_sampler(sim_powerlaw, params)

    for ind, sig in enumerate(sampler):
        assert isinstance(sig, np.ndarray)
    assert ind == 1
