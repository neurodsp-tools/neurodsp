"""Test combined simulation functions."""

from neurodsp.sim.combined import *

###################################################################################################
###################################################################################################

FS = 1000
N_SECONDS = 10

def test_sim_combined():

    simulations = {'sim_oscillation' : {'freq' : 10},
                   'sim_powerlaw' : {}}
    variances = [1, 0.5]
    out = sim_combined(N_SECONDS, FS, simulations, variances)
