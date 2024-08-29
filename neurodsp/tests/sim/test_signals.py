"""Tests for neurodsp.sim.signals."""

from pytest import raises

import numpy as np

from neurodsp.sim.signals import *

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
    assert sims_data.params is None
    assert sims_data.sim_func is None

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
    assert sims_data.params is None
    assert sims_data.sim_func is None

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

def test_multi_simulations():

    # Test empty initialization
    sims_empty = MultiSimulations()
    assert isinstance(sims_empty, MultiSimulations)

    # Demo data
    n_seconds = 2
    fs = 100
    n_sigs = 3
    n_sets = 2
    sigs = np.ones([n_sigs, n_seconds * fs])
    all_sigs = [sigs] * n_sets
    params = [{'n_seconds' : n_seconds, 'fs' : fs, 'exponent' : -2},
              {'n_seconds' : n_seconds, 'fs' : fs, 'exponent' : -1}]

    # Test initialization with data only
    sims_data = MultiSimulations(all_sigs)
    assert sims_data
    assert len(sims_data) == n_sets
    assert sims_data.n_seconds is None
    assert sims_data.fs is None
    assert sims_data.has_signals
    assert sims_data.params == [None] * n_sets
    assert sims_data.sim_func is None
    assert sims_data.values is None

    # Test dunders - iter & getitem & indicators
    for el in sims_data:
        assert isinstance(el, Simulations)
        assert len(el) == n_sigs
    assert isinstance(sims_data[0], Simulations)

    # Test initialization with metadata
    sims_full = MultiSimulations(all_sigs, params, 'sim_func', 'exponent')
    assert len(sims_full) == n_sets
    assert sims_full.has_signals
    for params_obj, params_org in zip(sims_full.params, params):
        assert params_obj == params_org
    assert sims_full.sim_func
    assert sims_full.values

def test_multi_simulations_add():

    sigs = [np.ones([2, 5]), np.ones([2, 5])]
    params = {'n_seconds' : 1, 'fs' : 100, 'param' : 'value'}

    sims_data1 = MultiSimulations(sigs)
    sims_data1.add_signals(sigs)
    assert sims_data1.has_signals

    sims_data2 = MultiSimulations(sigs, params)
    sims_data2.add_signals(sigs, params)
    assert sims_data2.has_signals
    assert len(sims_data2) == len(sims_data2.params)

    sims_data3 = MultiSimulations(sigs, params)
    sims_add = Simulations(sigs, params)
    sims_data3.add_signals(sims_add)
    assert sims_data3.has_signals

    sims_data4 = MultiSimulations(sigs, params)
    sims_data4.add_signals([sims_add, sims_add])
    assert sims_data4.has_signals
