"""Tests for neurodsp.utils.data."""

from numpy.testing import assert_equal

from neurodsp.tests.settings import FS, N_SECONDS

from neurodsp.utils.data import *

###################################################################################################
###################################################################################################

def test_create_freqs():

    freqs = create_freqs(8, 12)
    assert_equal(freqs, np.array([8, 9, 10, 11, 12]))

    freqs = create_freqs(8, 12, 0.5)
    assert_equal(freqs, np.array([8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]))

def test_create_times():

    times = create_times(N_SECONDS, FS)
    assert_equal(times, np.arange(0, N_SECONDS, 1/FS))

    start_val = 0.5
    times = create_times(N_SECONDS, FS, start_val=start_val)
    assert times[0] == start_val
    assert len(times) == N_SECONDS * FS

def test_create_samples():

    samples = create_samples(10)
    assert_equal(samples, np.arange(0, 10, 1))

def test_split_signal(tsig):

    chunks = split_signal(tsig, 100)
    assert chunks.shape == (10, 100)
    assert_equal(chunks[0, 0:10], tsig[0:10])
