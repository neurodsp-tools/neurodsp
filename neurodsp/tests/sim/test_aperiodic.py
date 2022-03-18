"""Tests for neurodsp.sim.aperiodic."""

import pytest

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit

from neurodsp.tests.settings import N_SECONDS, FS, EXP1, EXP2, KNEE, EPS
from neurodsp.tests.tutils import check_sim_output, check_exponent

from neurodsp.sim.aperiodic import *
from neurodsp.sim.aperiodic import _create_powerlaw
from neurodsp.spectral import compute_spectrum

###################################################################################################
###################################################################################################

def test_sim_poisson_pop():

    sig = sim_poisson_pop(N_SECONDS, FS)
    check_sim_output(sig)

def test_sim_synaptic_current():

    sig = sim_synaptic_current(N_SECONDS, FS)
    check_sim_output(sig)

def test_sim_knee():

    # Build the signal and run a smoke test
    sig = sim_knee(N_SECONDS, FS, EXP1, EXP2, KNEE)
    check_sim_output(sig, N_SECONDS, FS)

    # Check against the power spectrum when you take the Fourier transform
    sig_len = int(FS*N_SECONDS)
    freqs = np.linspace(0, FS/2, num=sig_len//2, endpoint=True)

    # Ignore the DC component to avoid division by zero in the Lorentzian
    freqs = freqs[1:]
    true_psd = 1 / ((freqs ** -EXP1 * (freqs ** (-EXP2 - EXP1)+ KNEE)))

    # Only look at the frequencies (ignoring DC component) up to the nyquist rate
    sig_hat = np.fft.fft(sig)[1:sig_len//2]
    numerical_psd = np.abs(sig_hat)**2

    scale = numerical_psd / true_psd
    np.allclose(true_psd*scale, numerical_psd, atol=EPS)

    # Accuracy test for a single exponent
    sig = sim_knee(N_SECONDS, FS, 0, EXP2, KNEE)

    freqs, powers = compute_spectrum(sig, FS, f_range=(1, 200))

    def _estimate_single_knee(xs, offset, knee, exponent):
        return np.zeros_like(xs) + offset - np.log10(xs**exponent + knee)

    ap_params, _ = curve_fit(_estimate_single_knee, freqs, np.log10(powers))
    _, KNEE_hat, EXP2_hat = ap_params[:]

    np.testing.assert_approx_equal(-EXP2_hat, EXP2, significant=1)
    np.testing.assert_approx_equal(KNEE_hat, KNEE, significant=1)

def test_sim_random_walk():

    sig = sim_random_walk(N_SECONDS, FS)
    check_sim_output(sig)

def test_sim_powerlaw():

    sig = sim_powerlaw(N_SECONDS, FS)
    check_sim_output(sig)

    # Test with a filter applied
    sig = sim_powerlaw(N_SECONDS, FS, f_range=(2, None))
    check_sim_output(sig)

@pytest.mark.parametrize('exponent', [-.5, 0, .5])
def test_sim_frac_gaussian_noise(exponent):

    # Simulate & check time series
    sig = sim_frac_gaussian_noise(N_SECONDS, FS, exponent=exponent)
    check_sim_output(sig)

    # Linear fit the log-log power spectrum & check error based on expected 1/f exponent
    freqs = np.linspace(1, FS//2, num=FS//2)
    powers = np.abs(np.fft.fft(sig)[1:FS//2 + 1]) ** 2
    [_, exponent_hat], _ = curve_fit(check_exponent, np.log10(freqs), np.log10(powers))
    assert abs(exponent_hat - exponent) < 0.2

@pytest.mark.parametrize('exponent', [-1.5, -2, -2.5])
def test_sim_frac_brownian_motion(exponent):

    # Simulate & check time series
    sig = sim_frac_brownian_motion(N_SECONDS, FS, exponent=exponent)
    check_sim_output(sig)

    # Linear fit the log-log power spectrum & check error based on expected 1/f exponent
    freqs = np.linspace(1, FS//2, num=FS//2)
    powers = np.abs(np.fft.fft(sig)[1:FS//2 + 1]) ** 2
    [_, exponent_hat], _ = curve_fit(check_exponent, np.log10(freqs), np.log10(powers))
    assert abs(exponent_hat - exponent) < 0.4

def test_create_powerlaw():

    sig = _create_powerlaw(int(N_SECONDS*FS), FS, -2)
    check_sim_output(sig)
