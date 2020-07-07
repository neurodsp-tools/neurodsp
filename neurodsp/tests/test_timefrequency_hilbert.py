"""Test functions for time-frequency Hilbert analyses."""

import numpy as np

from neurodsp.tests.settings import FS, EPS

from neurodsp.timefrequency.hilbert import *

from neurodsp.utils.data import create_times

###################################################################################################
###################################################################################################

def test_robust_hilbert(sig_sine):

    # Generate a signal with NaNs
    fs, n_points, n_nans = 100, 1000, 10
    sig = np.random.randn(n_points)
    sig[0:n_nans] = np.nan

    # Check has correct number of nans (not all nan), without increase_n
    hilb_sig = robust_hilbert(sig)
    assert sum(np.isnan(hilb_sig)) == n_nans

    # Check has correct number of nans (not all nan), with increase_n
    hilb_sig = robust_hilbert(sig, True)
    assert sum(np.isnan(hilb_sig)) == n_nans

    # Hilbert transform of sin(omega * t) = -sgn(omega) * cos(omega * t)
    n_seconds, fs, sig = sig_sine
    times = create_times(n_seconds, fs)

    # omega = 1
    hilbert_sig = np.imag(robust_hilbert(sig))
    answer = np.array([-np.cos(2*np.pi*time) for time in times])
    assert np.max(abs(hilbert_sig-answer)) < EPS

    # omega = -1
    sig = -sig
    hilbert_sig = np.imag(robust_hilbert(sig))
    answer = np.array([np.cos(2*np.pi*time) for time in times])
    assert np.max(abs(hilbert_sig-answer)) < EPS

def test_phase_by_time(tsig, sig_sine):

    out = phase_by_time(tsig, FS, (8, 12))
    assert out.shape == tsig.shape

    # Instantaneous phase of sin(t) should be piecewise linear with slope 1.
    n_seconds, fs, sig = sig_sine
    times = create_times(n_seconds, fs)
    # Scale the time axis to range over [0, 2pi].
    times = 2*np.pi*times

    phase = phase_by_time(sig, fs)
    answer = np.array([time-np.pi/2 if time <= 3*np.pi/2 else time-5*np.pi/2 for time in times])
    assert np.max(abs(answer-phase)) < EPS

def test_amp_by_time(tsig, sig_sine):

    out = amp_by_time(tsig, FS, (8, 12))
    assert out.shape == tsig.shape

    n_seconds, fs, sig = sig_sine
    times = create_times(n_seconds, fs)

    # Instantaneous amplitude of sinusoid should be 1 for all t.
    amp = amp_by_time(sig, fs)
    answer = np.array([1 for time in times])
    assert np.max(abs(answer-amp)) < EPS

def test_freq_by_time(tsig, sig_sine):

    out = freq_by_time(tsig, FS, (8, 12))
    assert out.shape == tsig.shape

    n_seconds, fs, sig = sig_sine
    times = create_times(n_seconds, fs)

    # Instantaneous frequency of sin(t) should be 1 for all t.
    freq = freq_by_time(sig, fs)
    answer = np.array([1 for time in times[1:]])
    assert np.max(abs(answer-freq[1:])) < EPS

def test_no_filters(tsig):

    out = phase_by_time(tsig, FS)
    assert out.shape == tsig.shape

    out = amp_by_time(tsig, FS)
    assert out.shape == tsig.shape

    out = freq_by_time(tsig, FS)
    assert out.shape == tsig.shape

def test_2d(tsig2d):

    out = phase_by_time(tsig2d, FS, (8, 12))
    assert out.shape == tsig2d.shape

    out = amp_by_time(tsig2d, FS, (8, 12))
    assert out.shape == tsig2d.shape

    out = freq_by_time(tsig2d, FS, (8, 12))
    assert out.shape == tsig2d.shape
