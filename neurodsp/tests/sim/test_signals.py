"""Tests for neurodsp.sim.signals."""

from pytest import raises

import numpy as np

from neurodsp.sim.params import get_base_params

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
    assert sims_data.function is None

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

    # Test pre-initialization
    sims_pre = Simulations(n_sigs, params, 'sim_func')
    assert len(sims_pre) == n_sigs
    assert np.sum(sims_pre.signals) == 0
    assert sims_pre.has_signals and sims_pre.has_params
    for ind, sig in enumerate(sigs):
        sims_pre.add_signal(sig, ind)
    assert len(sims_pre) == n_sigs
    assert np.sum(sims_pre.signals) != 0

def test_variable_simulations():

    # Test empty initialization
    sims_empty = VariableSimulations()
    assert isinstance(sims_empty, VariableSimulations)

    # Demo data
    n_seconds = 2
    fs = 100
    n_sigs = 2
    sigs = np.ones([2, n_seconds * fs])
    params = [{'n_seconds' : n_seconds, 'fs' : fs, 'exponent' : -2},
              {'n_seconds' : n_seconds, 'fs' : fs, 'exponent' : -1}]

    # Test initialization with data only
    sims_data = VariableSimulations(sigs)
    assert sims_data
    assert len(sims_data) == n_sigs
    assert sims_data.n_seconds is None
    assert sims_data.fs is None
    assert sims_data.has_signals
    assert sims_data.params is None
    assert sims_data.function is None

    # Test dunders - iter & getitem
    for el in sims_data:
        assert np.all(el)
    assert np.all(sims_data[0])

    # Test initialization with metadata
    sims_full = VariableSimulations(sigs, params, 'sim_func')
    assert len(sims_full) == n_sigs == len(sims_full.params)
    assert sims_full.params == params
    assert sims_full.has_signals
    assert sims_full.has_params

    # Test pre-initialization
    sims_pre = VariableSimulations(n_sigs, get_base_params(params), 'sim_func')
    assert len(sims_pre) == n_sigs
    assert np.sum(sims_pre.signals) == 0
    assert sims_pre.has_signals and sims_pre.has_params
    for ind, (sig, cparams) in enumerate(zip(sigs, params)):
        sims_pre.add_signal(sig, cparams, ind)
    assert len(sims_pre) == n_sigs
    assert np.sum(sims_pre.signals) != 0

def test_variable_simulations_add():

    sig = np.array([1, 2, 3, 4, 5])
    params = {'n_seconds' : 1, 'fs' : 100, 'param' : 'value'}
    sig2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    params2 = {'n_seconds' : 2, 'fs' : 250, 'param' : 'value'}

    sims_data1 = VariableSimulations(sig)
    sims_data1.add_signal(sig)
    assert sims_data1.has_signals

    sims_data2 = VariableSimulations(sig, params)
    sims_data2.add_signal(sig, params)
    assert sims_data2.has_signals
    assert sims_data2.has_params
    assert len(sims_data2) == len(sims_data2.params)

    ## ERROR CHECKS

    # Adding parameters with different base parameters
    sims_data3 = VariableSimulations(sig, params)
    with raises(ValueError):
        sims_data3.add_signal(sig2, params2)

    # Adding parameters without previous parameters
    sims_data4 = VariableSimulations(sig)
    with raises(ValueError):
        sims_data4.add_signal(sig, params)

    # Not adding parameters with previous parameters
    sims_data4 = VariableSimulations(sig, params)
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
    assert sims_data.function is None
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
    assert sims_full.function
    assert sims_full.values
    assert sims_full._base_params

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
