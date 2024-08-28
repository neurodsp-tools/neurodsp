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
    assert len(sims_data) == n_sigs
    assert sims_data.n_seconds is None
    assert sims_data.fs is None

    # Test initialization with metadata
    sims_full = Simulations(sigs, 'sim_func', params)
    assert len(sims_full) == n_sigs
    assert sims_full.params == params

def test_sampled_simulations():

    # Test empty initialization
    sims_empty = SampledSimulations()
    assert isinstance(sims_empty, SampledSimulations)

    # Demo data
    n_seconds = 2
    fs = 100
    n_sigs = 2
    sigs = np.zeros([2, n_seconds * fs])
    params = [{'n_seconds' : n_seconds, 'fs' : fs, 'exponent' : -2},
              {'n_seconds' : n_seconds, 'fs' : fs, 'exponent' : -1}]

    # Test initialization with data only
    sims_data = SampledSimulations(sigs)
    assert sims_data
    assert len(sims_data) == n_sigs
    assert sims_data.n_seconds is None
    assert sims_data.fs is None

    # Test initialization with metadata
    sims_full = SampledSimulations(sigs, 'sim_func', params)
    assert len(sims_full) == n_sigs == len(sims_full.params)
    assert sims_full.params == params
