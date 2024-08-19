"""Tests for neurodsp.sim.sims."""

import numpy as np

from neurodsp.sim.sims import *

###################################################################################################
###################################################################################################

def test_simulations():

    # Test empty initialization
    sims_empty = Simulations()
    assert isinstance(sims_empty, Simulations)

    # Demo data
    n_seconds = 2
    fs = 100
    n_sigs = 2
    sigs = np.zeros([2, n_seconds * fs])
    params = {'n_seconds' : n_seconds, 'fs' : fs, 'exponent' : -1}

    # Test initialization with data only
    sims_data = Simulations(sigs)
    assert sims_data

    # Test initialization with metadata
    sims_full = Simulations(sigs, 'sim_func', params)
    assert len(sims_full) == n_sigs
    assert sims_full.params == params
