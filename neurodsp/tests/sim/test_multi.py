"""Tests for neurodsp.sim.multi."""

import numpy as np

from neurodsp.sim.aperiodic import sim_powerlaw
from neurodsp.sim.update import create_updater, create_sampler, ParamSampler
from neurodsp.sim.signals import Simulations, VariableSimulations, MultiSimulations

from neurodsp.sim.multi import *

###################################################################################################
###################################################################################################

def test_sim_multiple():

    n_sims = 2
    params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}

    sims_obj = sim_multiple(sim_powerlaw, params, n_sims)
    assert isinstance(sims_obj, Simulations)
    assert sims_obj.signals.shape[0] == n_sims
    assert sims_obj.params == params

def test_sim_across_values(tsim_iters):

    params = [{'n_seconds' : 2, 'fs' : 250, 'exponent' : -2},
              {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}]

    sims_obj = sim_across_values(sim_powerlaw, params)
    assert isinstance(sims_obj, VariableSimulations)
    assert len(sims_obj) == len(params)
    for csim, cparams, oparams in zip(sims_obj, sims_obj.params, params):
        assert isinstance(csim, np.ndarray)
        assert cparams == oparams

    # Test with ParamIter input
    siter = tsim_iters['pl_exp']
    sims_iter = sim_across_values(sim_powerlaw, siter)
    assert isinstance(sims_iter, VariableSimulations)
    assert sims_iter.update == siter.update
    assert sims_iter.values == siter.values

def test_sim_multi_across_values(tsim_iters):

    n_sims = 3
    params = [{'n_seconds' : 2, 'fs' : 250, 'exponent' : -2},
              {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}]

    sims_obj = sim_multi_across_values(sim_powerlaw, params, n_sims)
    assert isinstance(sims_obj, MultiSimulations)
    for sims, cparams in zip(sims_obj, params):
        assert isinstance(sims, Simulations)
        assert len(sims) == n_sims
        assert sims.params == cparams

    # Test with ParamIter input
    siter = tsim_iters['pl_exp']
    sims_iter = sim_multi_across_values(sim_powerlaw, siter, n_sims)
    assert isinstance(sims_iter, MultiSimulations)
    assert sims_iter.update == siter.update
    assert sims_iter.values == siter.values

def test_sim_from_sampler():

    n_sims = 2
    params = {'n_seconds' : 10, 'fs' : 250, 'exponent' : None}
    samplers = {create_updater('exponent') : create_sampler([-2, -1, 0])}
    psampler = ParamSampler(params, samplers)

    sims_obj = sim_from_sampler(sim_powerlaw, psampler, n_sims)
    assert isinstance(sims_obj, VariableSimulations)
    assert sims_obj.signals.shape[0] == n_sims
    assert len(sims_obj.params) == n_sims
