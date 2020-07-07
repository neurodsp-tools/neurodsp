"""Configuration file for pytest for NDSP."""

import pytest

import numpy as np

from neurodsp.utils.sim import set_random_seed

from neurodsp.sim import sim_oscillation

###################################################################################################
###################################################################################################

def pytest_configure(config):

    set_random_seed(42)

@pytest.fixture(scope='session')
def tsig():

    yield np.random.randn(1000)

@pytest.fixture(scope='session')
def tsig2d():

    yield np.random.randn(2, 1000)

@pytest.fixture(scope='session')
def sig_sine():

	n_seconds = 1
	fs = 100
	yield (n_seconds, fs, sim_oscillation(n_seconds, fs, 1, variance=None, mean=None))