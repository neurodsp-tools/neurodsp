"""Test combined simulation functions."""

from neurodsp.sim.combined import *

###################################################################################################
###################################################################################################

FS = 100
N_SECONDS = 1

def test_sim_combined():

    simulations = {'sim_oscillation' : {'freq' : 10},
                   'sim_powerlaw' : {}}
    variances = [1, 0.5]
    out = sim_combined(N_SECONDS, FS, simulations, variances)

    # Test case with mutliple uses of same function
    simulations = {'sim_oscillation' : [{'freq' : 10}, {'freq' : 10}],
                   'sim_powerlaw' : {}}
    variances = [0.5, 0.5, 1]
    out = sim_combined(N_SECONDS, FS, simulations, variances)
