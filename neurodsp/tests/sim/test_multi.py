"""Tests for neurodsp.sim.multi."""

import numpy as np

from neurodsp.sim.aperiodic import sim_powerlaw
from neurodsp.sim.sims import Simulations
from neurodsp.sim.update import create_updater, create_sampler, ParamSampler

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

    params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}

    sims = sim_multiple(sim_powerlaw, params, 2, 'object')
    assert isinstance(sims, Simulations)
    assert sims.signals.shape[0] == 2
    assert sims.params == params

    sigs = sim_multiple(sim_powerlaw, params, 2, 'array')
    assert sigs.shape[0] == 2

def test_sim_across_values():

    params = [{'n_seconds' : 2, 'fs' : 250, 'exponent' : -2},
              {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}]
    sigs = sim_across_values(sim_powerlaw, params, 2)
    assert isinstance(sigs, dict)
    for val in [0, 1]:
        assert isinstance(sigs[0], np.ndarray)
        assert sigs[0].shape[0] == 2
    sigs_arr = sim_across_values(sim_powerlaw, params, 3, 'array')
    assert isinstance(sigs_arr, np.ndarray)
    assert sigs_arr.shape[0:2] == (2, 3)

def test_sim_from_sampler():

    params = {'n_seconds' : 10, 'fs' : 250, 'exponent' : None}
    samplers = {create_updater('exponent') : create_sampler([-2, -1, 0])}
    psampler = ParamSampler(params, samplers)

    sigs = sim_from_sampler(sim_powerlaw, psampler, 2)
    assert isinstance(sigs, np.ndarray)
    assert sigs.shape[0] == 2
