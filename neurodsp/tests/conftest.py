"""Configuration file for pytest for NDSP."""

import os
import shutil
import pytest

import numpy as np

from neurodsp.utils.sim import set_random_seed
from neurodsp.tests.settings import (FS, N_SECONDS, N_SECONDS_LONG, FREQ_SINE,
                                     BASE_TEST_FILE_PATH, TEST_PLOTS_PATH)

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

@pytest.fixture(scope='session', autouse=True)
def check_dir():
    """Once, prior to session, this will clear and re-initialize the test file directories."""

    # If the directories already exist, clear them
    if os.path.exists(BASE_TEST_FILE_PATH):
        shutil.rmtree(BASE_TEST_FILE_PATH)

    # Remake (empty) directories
    os.mkdir(BASE_TEST_FILE_PATH)
    os.mkdir(TEST_PLOTS_PATH)
