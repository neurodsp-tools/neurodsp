"""Configuration file for pytest for NDSP."""

import pytest

import numpy as np

from neurodsp.utils.sim import set_random_seed
from neurodsp.tests.settings import FS, N_SECONDS, N_SECONDS_LONG, FREQ_SINE

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
def tsig_sine():

	yield sim_oscillation(N_SECONDS, FS, freq=FREQ_SINE, variance=None, mean=None)

@pytest.fixture(scope='session')
def tsig_sine_long():

	yield sim_oscillation(N_SECONDS_LONG, FS, freq=FREQ_SINE, variance=None, mean=None)
