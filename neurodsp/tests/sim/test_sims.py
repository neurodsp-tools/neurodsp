"""Tests for neurodsp.sim.sims."""

from pytest import raises

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
    sigs = np.ones([2, n_seconds * fs])
    params = {'n_seconds' : n_seconds, 'fs' : fs, 'exponent' : -1}

    # Test initialization with data only
    sims_data = Simulations(sigs)
    assert sims_data
    assert len(sims_data) == n_sigs
    assert sims_data.n_seconds is None
    assert sims_data.fs is None
    assert sims_data.has_signals

    # Test dunders - iter & getitem & indicators
    for el in sims_data:
        assert np.all(el)
    assert np.all(sims_data[0])

    # Test initialization with metadata
    sims_full = Simulations(sigs, params, 'sim_func')
    assert len(sims_full) == n_sigs
    assert sims_full.params == params
    assert sims_full.has_signals
    assert sims_full.has_params

def test_sampled_simulations():

    # Test empty initialization
    sims_empty = SampledSimulations()
    assert isinstance(sims_empty, SampledSimulations)

    # Demo data
    n_seconds = 2
    fs = 100
    n_sigs = 2
    sigs = np.ones([2, n_seconds * fs])
    params = [{'n_seconds' : n_seconds, 'fs' : fs, 'exponent' : -2},
              {'n_seconds' : n_seconds, 'fs' : fs, 'exponent' : -1}]

    # Test initialization with data only
    sims_data = SampledSimulations(sigs)
    assert sims_data
    assert len(sims_data) == n_sigs
    assert sims_data.n_seconds is None
    assert sims_data.fs is None
    assert sims_data.has_signals

    # Test dunders - iter & getitem
    for el in sims_data:
        assert np.all(el)
    assert np.all(sims_data[0])

    # Test initialization with metadata
    sims_full = SampledSimulations(sigs, params, 'sim_func')
    assert len(sims_full) == n_sigs == len(sims_full.params)
    assert sims_full.params == params
    assert sims_full.has_signals
    assert sims_full.has_params

def test_sampled_simulations_add():

    sig = np.array([1, 2, 3, 4, 5])
    params = {'n_seconds' : 1, 'fs' : 100, 'param' : 'value'}
    sig2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    params2 = {'n_seconds' : 2, 'fs' : 250, 'param' : 'value'}

    sims_data1 = SampledSimulations(sig)
    sims_data1.add_signal(sig)
    assert sims_data1.has_signals

    sims_data2 = SampledSimulations(sig, params)
    sims_data2.add_signal(sig, params)
    assert sims_data2.has_signals
    assert sims_data2.has_params
    assert len(sims_data2) == len(sims_data2.params)

    ## ERROR CHECKS

    # Adding parameters with different base parameters
    sims_data3 = SampledSimulations(sig, params)
    with raises(ValueError):
        sims_data3.add_signal(sig2, params2)

    # Adding parameters without previous parameters
    sims_data4 = SampledSimulations(sig)
    with raises(ValueError):
        sims_data4.add_signal(sig, params)

    # Not adding parameters with previous parameters
    sims_data4 = SampledSimulations(sig, params)
    with raises(ValueError):
        sims_data4.add_signal(sig)
