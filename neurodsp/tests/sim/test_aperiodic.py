"""Test aperiodic simulation functions."""

from neurodsp.tests.settings import FS, N_SECONDS, N_SECONDS_LONG, EXP1, EXP2, KNEE, EPS
from neurodsp.tests.tutils import check_sim_output

from neurodsp.sim.aperiodic import *
from neurodsp.sim.aperiodic import _create_powerlaw

###################################################################################################
###################################################################################################

def test_sim_poisson_pop():

    sig = sim_poisson_pop(N_SECONDS, FS)
    check_sim_output(sig)

def test_sim_synaptic_current():

    sig = sim_synaptic_current(N_SECONDS, FS)
    check_sim_output(sig)

def test_sim_knee():

    # Use negative inputs for the exponents
    chi1 = EXP1
    chi2 = EXP2

    # Build the signal and run a smoke test
    sig = sim_knee(N_SECONDS, FS, chi1, chi2, KNEE)
    check_sim_output(sig)

    # Check against the power spectrum when you take the Fourier transform
    sig_len = int(FS*N_SECONDS)
    freqs = np.linspace(0, FS/2, num=sig_len//2, endpoint=True)

    # Ignore the DC component to avoid division by zero in the Lorentzian
    freqs = freqs[1:]
    true_psd = np.array([1/(f**-chi1*(f**(-chi2-chi1) + KNEE)) for f in freqs])

    # Only look at the frequencies (ignoring DC component) up to the nyquist rate
    sig_hat = np.fft.fft(sig)[1:sig_len//2]
    numerical_psd = np.abs(sig_hat)**2

    np.allclose(true_psd, numerical_psd, atol=EPS)

def test_sim_random_walk():

    sig = sim_random_walk(N_SECONDS, FS)
    check_sim_output(sig)

def test_sim_powerlaw():

    sig = sim_powerlaw(N_SECONDS, FS)
    check_sim_output(sig)

    # Test with a filter applied
    sig = sim_powerlaw(N_SECONDS, FS, f_range=(2, None))
    check_sim_output(sig)

def test_create_powerlaw():

    sig = _create_powerlaw(int(N_SECONDS*FS), FS, -2)
    check_sim_output(sig)
