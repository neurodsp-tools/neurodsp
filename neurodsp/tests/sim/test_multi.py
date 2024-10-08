"""Tests for neurodsp.sim.multi."""

import numpy as np

from neurodsp.sim.aperiodic import sim_powerlaw
from neurodsp.sim.update import create_updater, create_sampler, ParamSampler
from neurodsp.sim.signals import Simulations, SampledSimulations, MultiSimulations

from neurodsp.sim.multi import *

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

def test_sim_multiple():

    n_sims = 2
    params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}

    sims_obj = sim_multiple(sim_powerlaw, params, n_sims, 'object')
    assert isinstance(sims_obj, Simulations)
    assert sims_obj.signals.shape[0] == n_sims
    assert sims_obj.params == params

    sims_arr = sim_multiple(sim_powerlaw, params, n_sims, 'array')
    assert isinstance(sims_arr, np.ndarray)
    assert sims_arr.shape[0] == n_sims

def test_sim_across_values():

    n_sims = 3
    params = [{'n_seconds' : 2, 'fs' : 250, 'exponent' : -2},
              {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}]

    sims_obj = sim_across_values(sim_powerlaw, params, n_sims, 'object')
    assert isinstance(sims_obj, MultiSimulations)
    for sigs, cparams in zip(sims_obj, params):
        assert isinstance(sigs, Simulations)
        assert len(sigs) == n_sims
        assert sigs.params == cparams

    sigs_arr = sim_across_values(sim_powerlaw, params, n_sims, 'array')
    assert isinstance(sigs_arr, np.ndarray)
    assert sigs_arr.shape[0:2] == (len(params), n_sims)

def test_sim_from_sampler():

    n_sims = 2
    params = {'n_seconds' : 10, 'fs' : 250, 'exponent' : None}
    samplers = {create_updater('exponent') : create_sampler([-2, -1, 0])}
    psampler = ParamSampler(params, samplers)

    sims_obj = sim_from_sampler(sim_powerlaw, psampler, n_sims, 'object')
    assert isinstance(sims_obj, SampledSimulations)
    assert sims_obj.signals.shape[0] == n_sims
    assert len(sims_obj.params) == n_sims

    sims_arr = sim_from_sampler(sim_powerlaw, psampler, n_sims, 'array')
    assert isinstance(sims_arr, np.ndarray)
    assert sims_arr.shape[0] == n_sims
