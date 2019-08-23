"""Tests for data related utility functions."""

from numpy.testing import assert_equal

from neurodsp.utils.data import *

###################################################################################################
###################################################################################################

def test_create_freqs():

    freqs = create_freqs(8, 12)
    assert_equal(freqs, np.array([8, 9, 10, 11, 12]))

    freqs = create_freqs(8, 12, 0.5)
    assert_equal(freqs, np.array([8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]))

def test_create_times():

    times = create_times(1, 10)
    assert_equal(times, np.arange(0, 1, 1/10))

def test_create_samples():

    samples = create_samples(10)
    assert_equal(samples, np.arange(0, 10, 1))

def test_split_signal(tsig):

    chunks = split_signal(tsig, 100)
    assert chunks.shape == (10, 100)
    assert_equal(chunks[0, 0:10], tsig[0:10])
