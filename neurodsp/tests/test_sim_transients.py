"""Test transient simulation functions."""

from neurodsp.sim.transients import *

###################################################################################################
###################################################################################################

FS = 1000

def test_sim_make_synaptic_kernel():

    np.random.seed(0)

    # smoke test that valid parameter configurations do not return negative values
    t_ker, fs, tau_r, tau_d = 1., 1000., 0., 0.02
    assert np.all(sim_synaptic_kernel(t_ker, FS, tau_r, tau_d) >= 0.)

    t_ker, fs, tau_r, tau_d = 2., 2000., 0.005, 0.02
    assert np.all(sim_synaptic_kernel(t_ker, FS, tau_r, tau_d) >= 0.)

    t_ker, fs, tau_r, tau_d = 1., 1000., 0.02, 0.02
    assert np.all(sim_synaptic_kernel(t_ker, FS, tau_r, tau_d) >= 0.)

def test_sim_osc_cycle():
    pass
