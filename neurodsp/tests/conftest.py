"""Pytest configuration file for testing neurodsp."""

import os
import shutil
import pytest

import numpy as np

from neurodsp.sim import sim_oscillation, sim_powerlaw, sim_combined
from neurodsp.sim.update import create_updater, create_sampler
from neurodsp.sim.params import SimParams
from neurodsp.sim.signals import Simulations, VariableSimulations, MultiSimulations
from neurodsp.spectral import compute_spectrum
from neurodsp.utils.sim import set_random_seed
from neurodsp.tests.tsettings import (N_SECONDS, FS, FREQ_SINE, FREQ1, EXP1,
                                      BASE_TEST_FILE_PATH, TEST_PLOTS_PATH, TEST_FILES_PATH)

###################################################################################################
###################################################################################################

## TEST SETUP

def pytest_configure(config):

    set_random_seed(42)

@pytest.fixture(scope='session', autouse=True)
def check_dir():
    """Once, prior to session, this will clear and re-initialize the test file directories."""

    # If the directories already exist, clear them
    if os.path.exists(BASE_TEST_FILE_PATH):
        shutil.rmtree(BASE_TEST_FILE_PATH)

    # Remake (empty) directories
    os.mkdir(BASE_TEST_FILE_PATH)
    os.mkdir(TEST_PLOTS_PATH)
    os.mkdir(TEST_FILES_PATH)

## TEST OBJECTS

@pytest.fixture(scope='session')
def tsig():

    yield np.random.randn(1000)

@pytest.fixture(scope='session')
def tsig2d():

    yield np.random.randn(2, 1000)

@pytest.fixture(scope='session')
def tsig_sine():

    yield sim_oscillation(N_SECONDS, FS, freq=FREQ_SINE, variance=None, mean=None)

@pytest.fixture(scope='session')
def tsig_comb():

    components = {'sim_powerlaw': {'exponent' : EXP1},
                  'sim_oscillation': {'freq' : FREQ1}}
    yield sim_combined(n_seconds=N_SECONDS, fs=FS, components=components)

@pytest.fixture(scope='session')
def tsig_burst():

    components = {'sim_powerlaw': {'exponent' : EXP1},
                  'sim_bursty_oscillation': {'freq' : FREQ1}}
    yield sim_combined(n_seconds=N_SECONDS, fs=FS,
                       components=components, component_variances=[0.5, 1])

@pytest.fixture(scope='session')
def tspectrum(tsig_comb):

    freqs, powers = compute_spectrum(tsig_comb, FS)
    yield {'freqs' : freqs, 'powers' : powers}

@pytest.fixture(scope='session')
def tsim_params():

    sim_params = SimParams(N_SECONDS, FS)
    sim_params.register_group({
        'pl' : {'exponent' : -1},
        'osc' : {'freq' : -1},
    })

    yield sim_params

@pytest.fixture(scope='session')
def tsim_iters(tsim_params):

    sim_iters = tsim_params.to_iters()
    sim_iters.register_iter('pl_exp', 'pl', 'exponent', [-2, -1])

    yield sim_iters

@pytest.fixture(scope='session')
def tsim_samplers(tsim_params):

    sim_samplers = tsim_params.to_samplers()
    sim_samplers.register_sampler(\
        'samp_exp', 'pl', {create_updater('exponent') : create_sampler([-2, -1])})

    yield sim_samplers

@pytest.fixture(scope='session')
def tsims(tsig2d, tsim_params):

    yield Simulations(tsig2d, tsim_params['pl'], 'sim_test')

@pytest.fixture(scope='session')
def tvsims(tsig2d, tsim_iters):

    params = [ps for ps in tsim_iters.iters['pl_exp'].yielder]
    yield VariableSimulations(tsig2d, params, 'sim_test', tsim_iters.iters['pl_exp'].update)

@pytest.fixture(scope='session')
def tmsims(tsig2d, tsim_iters):

    params = [ps for ps in tsim_iters.iters['pl_exp'].yielder]
    yield MultiSimulations([tsig2d, tsig2d], params, 'sim_test', tsim_iters.iters['pl_exp'].update)
