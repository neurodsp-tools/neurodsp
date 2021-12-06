"""Tests for neurodsp.sim.periodic."""

from pytest import raises

from neurodsp.tests.tutils import check_sim_output
from neurodsp.tests.settings import FS, N_SECONDS, FREQ1, N_SECONDS_ODD, FS_ODD

from neurodsp.sim.periodic import *

###################################################################################################
###################################################################################################

def test_sim_oscillation():

    sig = sim_oscillation(N_SECONDS, FS, FREQ1)
    check_sim_output(sig)

    # Check some different frequencies, that they get expected length, etc
    for freq in [3.5, 7.0, 13.]:
        check_sim_output(sim_oscillation(N_SECONDS, FS, freq))

    # Check that nothing goes weird with different time & sampling rate inputs
    check_sim_output(sim_oscillation(N_SECONDS_ODD, FS, FREQ1), n_seconds=N_SECONDS_ODD)
    check_sim_output(sim_oscillation(N_SECONDS, FS_ODD, FREQ1), fs=FS_ODD)

def test_sim_oscillation_concatenation1():

    n_seconds, fs = 0.1, 1000

    # Test an odd frequency value, checking for a smooth concatenation
    freq = 13
    sig = sim_oscillation(n_seconds, fs, freq)

    # Test the concatenation is smooth - no large value differences
    assert np.all(np.abs(np.diff(sig)) < 2 * np.median(np.abs(np.diff(sig))))

    # Zoom in on concatenation point, checking for smooth transition
    concat_point = int(fs * 1/freq)
    vals = sig[concat_point-5:concat_point+5]
    assert np.all(np.diff(vals) > 0.5 * np.median(np.diff(vals)))
    assert np.all(np.diff(vals) < 1.5 * np.median(np.diff(vals)))

    # Test another, higher, frequency
    freq = 31
    sig = sim_oscillation(n_seconds, fs, freq)

    assert np.all(np.abs(np.diff(sig)) < 2 * np.median(np.abs(np.diff(sig))))

    concat_point = int(fs * 1/freq)
    vals = sig[concat_point-5:concat_point+5]
    assert np.all(np.diff(vals) > 0.5 * np.median(np.diff(vals)))
    assert np.all(np.diff(vals) < 1.5 * np.median(np.diff(vals)))

def test_sim_bursty_oscillation():

    # Check default values work
    sig1 = sim_bursty_oscillation(N_SECONDS, FS, FREQ1)
    check_sim_output(sig1)

    # Check probability burst approach
    sig2 = sim_bursty_oscillation(N_SECONDS, FS, FREQ1, \
        burst_def='prob', burst_params={'enter_burst' : 0.3, 'leave_burst' : 0.3})
    check_sim_output(sig2)

    # Check durations burst approach
    sig3 = sim_bursty_oscillation(N_SECONDS, FS, FREQ1, \
        burst_def='durations', burst_params={'n_cycles_burst' : 2, 'n_cycles_off' : 2})
    check_sim_output(sig3)


def test_sim_variable_oscillation():

    freqs = np.array([10, 20])
    rdsyms = [.4, .8]

    sig1 = sim_variable_oscillation(None, FS, freqs, cycle='asine', rdsym=rdsyms)
    assert isinstance(sig1, np.ndarray) and len(sig1) == sum(FS/freqs) and ~np.isnan(sig1).any()

    sig2 = sim_variable_oscillation(None, FS, 20, cycle='asine', rdsym=rdsyms)
    assert isinstance(sig2, np.ndarray) and len(sig2) == 2 * FS / 20 and ~np.isnan(sig2).any()

    # Too few frequencies
    with raises(ValueError):
        sig3 = sim_variable_oscillation(None, FS, freqs[1:], cycle='asine', rdsym=rdsyms)

    # Too few params
    with raises(ValueError):
        sig4 = sim_variable_oscillation(None, FS, freqs, cycle='asine', rdsym=rdsyms[1:])

def test_sim_damped_oscillation():

    sig1 = sim_damped_oscillation(N_SECONDS, FS, FREQ1, .1)
    sig2 = sim_damped_oscillation(N_SECONDS, FS, FREQ1, 50)
    sig3 = sim_damped_oscillation(N_SECONDS, FS, FREQ1, 25, 0.1)

    check_sim_output(sig1)
    check_sim_output(sig2)
    check_sim_output(sig3)

    # Large gammas range between (0, 1), whereas small gammas range between (-1, 1)
    assert sig1.sum() < sig2.sum()
    assert len(sig1) == len(sig2)

def test_make_bursts():

    is_osc = np.array([False, False, True, True, False, True, False, True, True, False])
    cycle = np.ones([10])

    sig = make_bursts(N_SECONDS, FS, is_osc, cycle)
    check_sim_output(sig)

    # Test make bursts with uneven division of signal and cycle divisions
    #   In this test, there aren't enough samples in the signal to add last cycle
    is_osc = np.array([False, True, True])
    cycle = np.ones([7])

    sig = make_bursts(2, 10, is_osc, cycle)
    assert sum(sig) > 0

def test_make_is_osc_prob():

    is_osc = make_is_osc_prob(15, 0.5, 0.5)
    assert is_osc.dtype == 'bool'
    assert sum(is_osc) < len(is_osc)

def test_make_is_osc_durations():

    is_osc = make_is_osc_durations(15, 2, 2)
    assert is_osc.dtype == 'bool'
    assert list(is_osc[0:2]) == [True, True]
    assert list(is_osc[2:4]) == [False, False]

def test_get_burst_samples():

    is_oscillating = np.array([False, True, True, False])
    burst_samples = get_burst_samples(is_oscillating, FS, 10)

    # First ten samples should be false & next ten samples should by true
    assert sum(burst_samples[0:10]) == 0
    assert sum(burst_samples[10:20]) == 10
