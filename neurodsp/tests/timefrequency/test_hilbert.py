"""Tests for neurodsp.timefrequency.hilbert."""

import numpy as np

from neurodsp.tests.settings import FS, N_SECONDS, F_RANGE, EPS

from neurodsp.timefrequency.hilbert import *

from neurodsp.utils.data import create_times

###################################################################################################
###################################################################################################

def test_robust_hilbert(tsig_sine):

    # Generate a signal with NaNs
    n_nans = 10
    sig = np.random.randn(int(FS*N_SECONDS))
    sig[0:n_nans] = np.nan

    # Check has correct number of nans (not all nan)
    hilb_sig = robust_hilbert(sig)
    assert sum(np.isnan(hilb_sig)) == n_nans

    times = create_times(N_SECONDS, FS)

    # Hilbert transform of sin(omega * t) = -sign(omega) * cos(omega * t)

    # omega = 1
    hilbert_sig = np.imag(robust_hilbert(tsig_sine))
    expected_answer = np.array([-np.cos(2*np.pi*time) for time in times])
    assert np.allclose(hilbert_sig, expected_answer, atol=EPS)

    # omega = -1
    hilbert_sig = np.imag(robust_hilbert(-tsig_sine))
    expected_answer = np.array([np.cos(2*np.pi*time) for time in times])
    assert np.allclose(hilbert_sig, expected_answer, atol=EPS)

def test_phase_by_time(tsig, tsig_sine):

    # Check that a random signal, with a filter applied, runs & preserves shape
    out = phase_by_time(tsig, FS, F_RANGE)
    assert out.shape == tsig.shape

    # Check the expected answer for the test sine wave signal
    #   The instantaneous phase of sin(t) should be piecewise linear with slope 1
    phase = phase_by_time(tsig_sine, FS)

    # Check the first second of the signal, and create associated time axis, scaled to [0, 2pi]
    phase = phase[0:int(FS*1.0)]
    times = 2 * np.pi * create_times(1.0, FS)

    # Generate the expected instantaneous phase of the given signal
    #   Phase is defined in [-pi, pi]. Since sin(t) = cos(t - pi/2), the phase should begin at
    #   -pi/2 and increase with a slope of 1 until phase hits pi, or when t=3pi/2. Phase then
    #   wraps around to -pi and again increases linearly with a slope of 1
    expected_answer = np.array(\
        [time-np.pi/2 if time <= 3*np.pi/2 else time-5*np.pi/2 for time in times])

    # Round to the sixth decimal place and convert from (-pi, pi) to (0, 2pi)
    expected_answer = np.mod(expected_answer.round(6), 2*np.pi)
    phase = np.mod(phase.round(6), 2*np.pi)

    assert np.allclose(expected_answer, phase, atol=EPS)

def test_amp_by_time(tsig, tsig_sine):

    # Check that a random signal, with a filter applied, runs & preserves shape
    out = amp_by_time(tsig, FS, F_RANGE)
    assert out.shape == tsig.shape

    # Instantaneous amplitude of sinusoid should be 1 for all timepoints
    amp = amp_by_time(tsig_sine, FS)
    expected_answer = np.ones_like(amp)

    assert np.allclose(expected_answer, amp, atol=EPS)

def test_freq_by_time(tsig, tsig_sine):

    # Check that a random signal, with a filter applied, runs & preserves shape
    out = freq_by_time(tsig, FS, F_RANGE)
    assert out.shape == tsig.shape

    # Instantaneous frequency of sin(t) should be 1 for all timepoints
    freq = freq_by_time(tsig_sine, FS)
    expected_answer = np.ones_like(freq)

    assert np.allclose(expected_answer[1:], freq[1:], atol=EPS)

def test_2d(tsig2d):

    out = phase_by_time(tsig2d, FS, F_RANGE)
    assert out.shape == tsig2d.shape

    out = amp_by_time(tsig2d, FS, F_RANGE)
    assert out.shape == tsig2d.shape

    out = freq_by_time(tsig2d, FS, F_RANGE)
    assert out.shape == tsig2d.shape
