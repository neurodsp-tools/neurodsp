"""Tests for simulation info functions."""

from pytest import raises

from neurodsp.sim.info import *

###################################################################################################
###################################################################################################

def test_get_sim_funcs():

    for module in SIM_MODULES:
        funcs = get_sim_funcs(module)
        assert isinstance(funcs, dict)

    # Check the error for requesting non-existing function
    with raises(ValueError):
        get_sim_func('bad_mod')

def test_get_sim_names():

    for module in SIM_MODULES:
        funcs = get_sim_names(module)
        assert isinstance(funcs, list)

def test_get_sim_func():

    # Check a successful function request
    func = get_sim_func('sim_oscillation')

    # Check the error for requesting non-existing function
    with raises(ValueError):
        get_sim_func('bad_func')
