"""Test combined simulation functions."""

from pytest import raises

from neurodsp.tests.settings import FS, N_SECONDS, FREQ1, FREQ2
from neurodsp.tests.tutils import check_sim_output

from neurodsp.sim.combined import *

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
