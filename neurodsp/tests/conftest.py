"""Configuration file for pytest for NDSP."""

import pytest

from neurodsp.tests.utils import get_random_signal

###################################################################################################
###################################################################################################

@pytest.fixture(scope='session')
def tsig():
    yield get_random_signal()
