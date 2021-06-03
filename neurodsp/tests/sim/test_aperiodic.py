"""Tests for neurodsp.sim.aperiodic."""

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
    check_sim_output(sig)

    # Check against the power spectrum when you take the Fourier transform
    sig_len = int(FS*N_SECONDS)
    freqs = np.linspace(0, FS/2, num=sig_len//2, endpoint=True)

    # Ignore the DC component to avoid division by zero in the Lorentzian
    freqs = freqs[1:]
    true_psd = np.array([1/(freq**-EXP1*(freq**(-EXP2-EXP1) + KNEE)) for freq in freqs])

    # Only look at the frequencies (ignoring DC component) up to the nyquist rate
    sig_hat = np.fft.fft(sig)[1:sig_len//2]
    numerical_psd = np.abs(sig_hat)**2

    np.allclose(true_psd, numerical_psd, atol=EPS)

    # Accuracy test for a single exponent
    sig = sim_knee(n_seconds=N_SECONDS, fs=FS, chi1=0, chi2=EXP2, knee=KNEE)

    freqs, powers = compute_spectrum(sig, FS, f_range=(1, 200))

    def _estimate_single_knee(xs, offset, knee, exponent):
        return np.zeros_like(xs) + offset - np.log10(xs**exponent + knee)

    ap_params, _ = curve_fit(_estimate_single_knee, freqs, np.log10(powers))
    _, _, EXP2_hat = ap_params[:]

    assert -round(EXP2_hat) == EXP2

def test_sim_random_walk():

    sig = sim_random_walk(N_SECONDS, FS)
    check_sim_output(sig)

def test_sim_powerlaw():

    sig = sim_powerlaw(N_SECONDS, FS)
    check_sim_output(sig)

    # Test with a filter applied
    sig = sim_powerlaw(N_SECONDS, FS, f_range=(2, None))
    check_sim_output(sig)

def test_sim_frac_gaussian_noise():

    chis = np.array([-.5, 0, .5])
    freqs = np.linspace(1, FS//2, num=FS//2)
    error = np.zeros_like(chis)

    for idx, chi in enumerate(chis):

        # Simulate
        sig = sim_frac_gaussian_noise(N_SECONDS, FS, chi=chi)
        powers = np.abs(np.fft.fft(sig)[1:FS//2+1])**2

        # Linear fit in log-log
        [_, chi_hat], _ = curve_fit(check_exponent, np.log10(freqs), np.log10(powers))

        # Compute error
        error[idx] = abs(chi_hat - chi)

    # Ensure mean error is less than 0.2 exponent
    assert np.mean(error) < 0.2

def test_sim_frac_brownian_motion():

    chis = np.array([-1.5, -2, -2.5])
    freqs = np.linspace(1, FS//2, num=FS//2)
    error = np.zeros_like(chis)

    for idx, chi in enumerate(chis):

        # Simulate
        sig = sim_frac_brownian_motion(N_SECONDS, FS, chi=chi)
        powers = np.abs(np.fft.fft(sig)[1:FS//2+1])**2

        # Linear fit in log-log
        [_, chi_hat], _ = curve_fit(check_exponent, np.log10(freqs), np.log10(powers))

        # Compute error
        error[idx] = abs(chi_hat - chi)

    # Ensure mean error less than 0.4 exponent
    assert np.mean(error) < 0.4

def test_create_powerlaw():

    sig = _create_powerlaw(int(N_SECONDS*FS), FS, -2)
    check_sim_output(sig)
