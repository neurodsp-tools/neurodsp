"""Tests for neurodsp.sim.combined."""

from pytest import raises

from neurodsp.tests.settings import FS, N_SECONDS, FREQ1, FREQ2, EXP1
from neurodsp.tests.tutils import check_sim_output

from neurodsp.sim.combined import *
from neurodsp.sim import sim_powerlaw
from neurodsp.spectral import compute_spectrum

###################################################################################################
###################################################################################################

def test_sim_combined():

    simulations = {'sim_oscillation' : {'freq' : FREQ1},
                   'sim_powerlaw' : {'exponent' : -2}}

    out = sim_combined(N_SECONDS, FS, simulations)
    check_sim_output(out)

    # Test case with multiple uses of same function
    simulations = {'sim_oscillation' : [{'freq' : FREQ1}, {'freq' : FREQ2}],
                   'sim_powerlaw' : {'exponent' : -2}}
    variances = [0.5, 0.5, 1]
    out = sim_combined(N_SECONDS, FS, simulations, variances)
    check_sim_output(out)

    # Check the variance mismatch error
    variances = [0.5, 1]
    with raises(ValueError):
        out = sim_combined(N_SECONDS, FS, simulations, variances)

def test_sim_peak_oscillation():

    sig_ap = sim_powerlaw(N_SECONDS, FS)
    sig = sim_peak_oscillation(sig_ap, FS, FREQ1, bw=5, height=10)

    check_sim_output(sig)

    # Ensure the peak is at or close (+/- 5hz) to FREQ1
    _, powers_ap = compute_spectrum(sig_ap, FS)
    _, powers = compute_spectrum(sig, FS)

    assert abs(np.argmax(powers-powers_ap) - FREQ1) < 5

def test_sim_modulated_signal():

    msig1 = sim_modulated_signal(N_SECONDS, FS,
                                'sim_oscillation', {'freq' : 10},
                                'sim_oscillation', {'freq' : 1})
    check_sim_output(msig1)

    msig2 = sim_modulated_signal(N_SECONDS, FS,
                                 'sim_oscillation', {'freq' : 10},
                                 'sim_powerlaw', {})
    check_sim_output(msig2)
