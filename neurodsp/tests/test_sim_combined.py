"""Test combined simulation functions."""

from pytest import raises

from neurodsp.tests.utils import FS, N_SECONDS, FREQ
from neurodsp.tests.utils import check_sim_output

from neurodsp.sim.combined import *

###################################################################################################
###################################################################################################

def test_sim_combined():

    simulations = {'sim_oscillation' : {'freq' : FREQ},
                   'sim_powerlaw' : {'exponent' : -2}}

    out = sim_combined(N_SECONDS, FS, simulations)
    check_sim_output(out)

    # Test case with mutliple uses of same function
    simulations = {'sim_oscillation' : [{'freq' : FREQ}, {'freq' : 20}],
                   'sim_powerlaw' : {'exponent' : -2}}
    variances = [0.5, 0.5, 1]
    out = sim_combined(N_SECONDS, FS, simulations, variances)
    check_sim_output(out)

    # Check the variance mismatch error
    variances = [0.5, 1]
    with raises(ValueError):
        out = sim_combined(N_SECONDS, FS, simulations, variances)
