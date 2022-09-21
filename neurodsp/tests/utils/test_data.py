"""Tests for neurodsp.utils.data."""

from numpy.testing import assert_equal, assert_almost_equal

from neurodsp.tests.settings import N_SECONDS, FS, N_SECONDS_ODD, FS_ODD

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
    assert len(times) == compute_nsamples(N_SECONDS, FS)
    assert_equal(times, np.arange(0, N_SECONDS, 1/FS))

    start_val = 0.5
    times = create_times(N_SECONDS, FS, start_val=start_val)
    assert_equal(times[0], start_val)
    assert_almost_equal(times[-1], N_SECONDS + start_val, decimal=2)
    assert len(times) == compute_nsamples(N_SECONDS, FS)

    assert len(create_times(N_SECONDS_ODD, FS)) == compute_nsamples(N_SECONDS_ODD, FS)
    assert len(create_times(N_SECONDS, FS_ODD)) == compute_nsamples(N_SECONDS, FS_ODD)
    assert len(create_times(N_SECONDS_ODD, FS_ODD)) == compute_nsamples(N_SECONDS_ODD, FS_ODD)

def test_create_samples():

    samples = create_samples(10)
    assert_equal(samples, np.arange(0, 10, 1))

def test_compute_nsamples():

    n_samples = compute_nsamples(N_SECONDS, FS)
    assert isinstance(n_samples, int)
    assert n_samples == 1000

    n_samples = compute_nsamples(N_SECONDS_ODD, FS_ODD)
    assert isinstance(n_samples, int)
    assert n_samples == int(np.ceil(N_SECONDS_ODD * FS_ODD))

def test_compute_nseconds(tsig):

    n_seconds = compute_nseconds(tsig, FS)
    assert isinstance(n_seconds, float)
    assert n_seconds == N_SECONDS

def test_compute_cycle_nseconds():

    n_seconds10 = compute_cycle_nseconds(10)
    assert isinstance(n_seconds10, float)
    assert n_seconds10 == 0.1

    n_seconds10fs = compute_cycle_nseconds(10, fs=1001)
    assert isinstance(n_seconds10fs, float)
    assert n_seconds10fs != n_seconds10

def test_split_signal(tsig):

    chunks = split_signal(tsig, 100)
    assert chunks.shape == (10, 100)
    assert_equal(chunks[0, 0:10], tsig[0:10])
