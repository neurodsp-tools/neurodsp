"""Test aperiodic simulation functions."""

from neurodsp.tests.settings import FS, FS_HIGH, N_SECONDS
from neurodsp.tests.tutils import check_sim_output, check_exponent

from neurodsp.sim.aperiodic import *
from neurodsp.sim.aperiodic import _create_powerlaw

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit

###################################################################################################
###################################################################################################

def test_sim_poisson_pop():

    sig = sim_poisson_pop(N_SECONDS, FS)
    check_sim_output(sig)

def test_sim_synaptic_current():

    sig = sim_synaptic_current(N_SECONDS, FS)
    check_sim_output(sig)

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
    freqs = np.linspace(1, FS_HIGH//2, num=FS_HIGH//2)
    error = np.zeros_like(chis)

    for idx, chi in enumerate(chis):

        # Simulate
        sig = sim_frac_gaussian_noise(N_SECONDS, FS_HIGH, chi=chi)
        powers = np.abs(np.fft.fft(sig)[1:FS_HIGH//2+1])**2

        # Linear fit in log-log
        [_, chi_hat], _ = curve_fit(check_exponent, np.log10(freqs), np.log10(powers))

        # Compute error
        error[idx] = abs(chi_hat - chi)

    # Ensure mean error is less than 0.2 exponent
    assert np.mean(error) < 0.2

def test_sim_frac_brownian_motion():

    chis = np.array([-1.5, -2, -2.5])
    freqs = np.linspace(1, FS_HIGH//2, num=FS_HIGH//2)
    error = np.zeros_like(chis)

    for idx, chi in enumerate(chis):

        # Simulate
        sig = sim_frac_brownian_motion(N_SECONDS, FS_HIGH, chi=chi)
        powers = np.abs(np.fft.fft(sig)[1:FS_HIGH//2+1])**2

        # Linear fit in log-log
        [_, chi_hat], _ = curve_fit(check_exponent, np.log10(freqs), np.log10(powers))

        # Compute error
        error[idx] = abs(chi_hat - chi)

    # Ensure mean error less than 0.4 exponent
    assert np.mean(error) < 0.4

def test_create_powerlaw():

    sig = _create_powerlaw(int(N_SECONDS*FS), FS, -2)
    check_sim_output(sig)
