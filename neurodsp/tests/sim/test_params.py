"""Tests for neurodsp.sim.params."""

from neurodsp.sim.update import create_updater, create_sampler

from neurodsp.sim.params import *

###################################################################################################
###################################################################################################

def test_sim_params():

    # Test initialization
    sps1 = SimParams(5, 250)
    assert sps1

    # Define components to add
    comp1 = {'sim_powerlaw' : {'exponent' : -1}}
    comp2 = {'sim_oscillation' : {'freq' : -1}}

    # Test registering new simulation parameter definition
    sps1.register('pl', comp1)
    assert comp1.items() <= sps1['pl'].items()

    # Test registering a group of new simulation parameter definitions
    sps2 = SimParams(5, 250)
    sps2.register_group({'pl' : comp1, 'osc' : comp2})
    assert comp1.items() <= sps2['pl'].items()
    assert comp2.items() <= sps2['osc'].items()

def test_sim_params_to(tsim_params):
    # Test the SimParams `to_` extraction functions

    iters = tsim_params.to_iters()
    assert iters.base == tsim_params.base
    assert tsim_params['pl'] == iters.params['pl']
    assert tsim_params['osc'] == iters.params['osc']

    samplers = tsim_params.to_samplers(n_samples=10)
    assert samplers.base == tsim_params.base
    assert tsim_params['pl'] == samplers.params['pl']
    assert tsim_params['osc'] == samplers.params['osc']

def test_sim_iters():

    comp_plw = {'sim_powerlaw' : {'exponent' : -1}}
    comp_osc = {'sim_oscillation' : {'freq' : -1}}

    sis1 = SimIters(5, 250)
    sis1.register('pl', comp_plw)
    sis1.register_iter('pl_exp', 'pl', 'exponent', [-2, -1, 0])
    assert sis1['pl_exp']
    assert sis1['pl_exp'].values == [-2, -1, 0]

    # Test registering a group of new simulation iterator definitions
    sis2 = SimIters(5, 250)
    sis2.register_group({'pl' : comp_plw, 'osc' : comp_osc})
    sis2.register_group_iters([
        {'name' : 'pl_exp', 'label' : 'pl', 'update' : 'exponent', 'values' : [-2, -1 ,0]},
        {'name' : 'osc_freq', 'label' : 'osc', 'update' : 'freq', 'values' : [10, 20, 30]},
    ])
    assert sis2['pl_exp']
    assert sis2['osc_freq']

def test_sim_samplers():

    sss1 = SimSamplers(5, 250)
    sss1.register('pl', {'sim_powerlaw' : {'exponent' : -1}})
    sss1.register_sampler('samp_exp', 'pl', {create_updater('exponent') : create_sampler([-2, -1, 0])})
    assert sss1['samp_exp'] is not None

    # Test registering a group of new simulation sampler definitions
    sss2 = SimSamplers(5, 250)
    sss2.register_group({
        'pl' : {'sim_powerlaw' : {'exponent' : -1}},
        'osc' : {'sim_oscillation' : {'freq' : -1}},
    })
    sss2.register_group_samplers([
        {'name' : 'samp_exp', 'label' : 'pl',
         'samplers' : {create_updater('exponent') : create_sampler([-2, -1, 0])}},
        {'name' : 'samp_freq', 'label' : 'osc',
         'samplers' : {create_updater('freq') : create_sampler([10, 20, 30])}},
    ])
    assert sss2['samp_exp'] is not None
    assert sss2['samp_freq'] is not None
