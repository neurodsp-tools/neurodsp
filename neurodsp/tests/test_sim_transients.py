"""Test transient simulation functions."""

from pytest import raises

from neurodsp.sim.transients import *

###################################################################################################
###################################################################################################

N_SECONDS = 1
FS = 100

def test_sim_osc_cycle():

    cycle = sim_osc_cycle(N_SECONDS, FS, 'sine')
    cycle = sim_osc_cycle(N_SECONDS, FS, 'asine', rdsym=0.75)
    cycle = sim_osc_cycle(N_SECONDS, FS, 'sawtooth', width=0.5)
    cycle = sim_osc_cycle(N_SECONDS, FS, 'gaussian', std=2)
    cycle = sim_osc_cycle(N_SECONDS, FS, 'exp', tau_d=0.2)
    cycle = sim_osc_cycle(N_SECONDS, FS, '2exp', tau_r=0.2, tau_d=0.2)

    with raises(ValueError):
        sim_osc_cycle(N_SECONDS, FS, 'not_a_cycle')

def test_asine_cycle():

    cycle = sim_asine_cycle(N_SECONDS, FS, 0.25)
    assert True

def test_sim_make_synaptic_kernel():

    np.random.seed(0)

    # smoke test that valid parameter configurations do not return negative values
    t_ker, tau_r, tau_d = 1., 0., 0.02
    assert np.all(sim_synaptic_kernel(t_ker, FS, tau_r, tau_d) >= 0.)

    t_ker, tau_r, tau_d = 2., 0.005, 0.02
    assert np.all(sim_synaptic_kernel(t_ker, FS, tau_r, tau_d) >= 0.)

    t_ker, tau_r, tau_d = 1., 0.02, 0.02
    assert np.all(sim_synaptic_kernel(t_ker, FS, tau_r, tau_d) >= 0.)

def test_create_cycle_time():

    times = create_cycle_time(N_SECONDS, FS)
    assert True
