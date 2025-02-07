"""Tests for neurodsp.sim.params."""

from pytest import raises

from neurodsp.sim.update import create_updater, create_sampler

from neurodsp.sim.params import *

###################################################################################################
###################################################################################################

## FUNCTION TESTS

def test_get_base_params(tsim_iters):

    params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}
    out1 = get_base_params(params)
    for bparam in out1:
        assert bparam in BASE_PARAMS

    params_lst = [params, params]
    out2 = get_base_params(params)
    for bparam in out2:
        assert bparam in BASE_PARAMS

    out3 = get_base_params(tsim_iters['pl_exp'])
    for bparam in out3:
        assert bparam in BASE_PARAMS

    with raises(ValueError):
        get_base_params('parameters')

def test_drop_base_params():

    params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}
    out1 = drop_base_params(params)
    for bparam in BASE_PARAMS:
        assert bparam not in out1
    assert 'exponent' in out1

    params_lst = [params, params]
    out2 = drop_base_params(params_lst)
    for cparams in out2:
        for bparam in BASE_PARAMS:
            assert bparam not in cparams
        assert 'exponent' in cparams

    with raises(ValueError):
        drop_base_params('parameters')

def test_get_param_values():

    params = [{'n_seconds' : 2, 'fs' : 250, 'exponent' : -2},
              {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}]
    assert get_param_values(params, 'exponent') == [-2, -1]
    assert get_param_values(params, 'n_seconds') == [2, 2]

    params = [{'n_seconds' : 2, 'fs' : 250, 'components' : \
                {'sim_powerlaw' : {'exponent' : -2}, 'sim_oscillation' : {'freq' : 10}}},
              {'n_seconds' : 2, 'fs' : 250, 'components' : \
                {'sim_powerlaw' : {'exponent' : -1}, 'sim_oscillation' : {'freq' : 10}}}]
    assert get_param_values(params, 'exponent', 'sim_powerlaw') == [-2, -1]
    assert get_param_values(params, 'freq', 'sim_oscillation') == [10, 10]

## CLASS TESTS

def test_sim_params():

    # Test initialization
    sps1 = SimParams(5, 250)
    assert sps1

    # Define components to add
    comp1 = {'sim_powerlaw' : {'exponent' : -1}}
    comp2 = {'sim_oscillation' : {'freq' : -1}}

    # Test registering new simulation parameter definition
    sps1.register('pl', comp1)
    assert 'pl' in sps1
    assert comp1.items() <= sps1['pl'].items()

    # Test registering a group of new simulation parameter definitions
    sps2 = SimParams(5, 250)
    sps2.register_group({'pl' : comp1, 'osc' : comp2})
    for label in ['pl', 'osc']:
        assert label in sps2
    assert comp1.items() <= sps2['pl'].items()
    assert comp2.items() <= sps2['osc'].items()

    # Test clearing and re-registering
    sps2.register_group({'pl2' : comp1, 'osc2' : comp2}, clear=True)
    for old_label in ['pl', 'osc']:
        with raises(KeyError):
            sps2[old_label]
    for label in ['pl2', 'osc2']:
        assert label in sps2
    assert comp1.items() <= sps2['pl2'].items()
    assert comp2.items() <= sps2['osc2'].items()

def test_sim_params_props(tsim_params):

    # Test properties
    assert tsim_params.labels
    assert tsim_params.params

    # Test copy and clear
    ntsim = tsim_params.copy()
    assert ntsim != tsim_params
    ntsim.clear(True)
    assert ntsim.params == {}
    assert ntsim.base == {'n_seconds': None, 'fs': None}

def test_sim_params_make_params(tsim_params):
    # Test the SimParams `make_` methods

    # Operate on a copy
    ntsim = tsim_params.copy()
    ntsim.clear()

    out1 = ntsim.make_params({'exponent' : -1})
    assert isinstance(out1, dict)
    assert out1['n_seconds'] == ntsim.n_seconds
    assert out1['exponent'] == -1

    out2 = ntsim.make_params({'exponent' : -1}, f_range=(1, 50))
    assert out2['f_range'] == (1, 50)

    comps = [{'sim_powerlaw' : {'exponent' : -1}, 'sim_oscillation' : {'freq' : 10}}]
    out3 = ntsim.make_params(comps)
    assert out3['components'] == comps[0]

def test_sim_params_upd(tsim_params):
    # Test the SimParams `update_` methods

    # Operate on a copy
    ntsim = tsim_params.copy()

    # Update base
    ntsim.update_base(123, 123)
    assert ntsim.n_seconds == 123
    assert ntsim.fs == 123

    # Update param
    ntsim.update_param('pl', 'sim_powerlaw', {'exponent' : -3})
    assert ntsim.params['pl']['sim_powerlaw']['exponent'] == -3

def test_sim_params_to(tsim_params):
    # Test the SimParams `to_` extraction methods

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
    assert 'pl_exp' in sis1
    assert sis1['pl_exp'].values == [-2, -1, 0]

    # Test registering a group of new simulation iterator definitions
    sis2 = SimIters(5, 250)
    sis2.register_group({'pl' : comp_plw, 'osc' : comp_osc})
    sis2.register_group_iters([
        {'name' : 'pl_exp', 'label' : 'pl', 'update' : 'exponent', 'values' : [-2, -1 ,0]},
        {'name' : 'osc_freq', 'label' : 'osc', 'update' : 'freq', 'values' : [10, 20, 30]},
    ])
    for label in ['pl_exp', 'osc_freq']:
        assert label in sis2

    # Test clearing and re-registering group with list inputs
    sis2.register_group_iters([
        ['pl_exp2', 'pl', 'exponent', [-2, -1 ,0]],
        ['osc_freq2', 'osc', 'freq', [10, 20, 30]],
    ], clear=True)
    for old_label in ['pl_exp', 'osc_freq']:
        with raises(KeyError):
            sis2[old_label]
    for label in ['pl_exp2', 'osc_freq2']:
        assert label in sis2

def test_sim_iters_props(tsim_iters):

    # Test properties
    assert tsim_iters.labels
    assert tsim_iters.iters

    # Test copy and clear
    ntiter = tsim_iters.copy()
    assert ntiter != tsim_iters
    ntiter.clear(True, True, True)
    assert ntiter.iters == {}
    assert ntiter.params == {}
    assert ntiter.base == {'n_seconds': None, 'fs': None}

def test_sim_iters_upd(tsim_iters):

    tsim_iters.update_iter('pl_exp', 'values', [-3, -2, -1])
    assert tsim_iters.iters['pl_exp'].values == [-3, -2, -1]

def test_sim_samplers():

    sss1 = SimSamplers(5, 250)
    sss1.register('pl', {'sim_powerlaw' : {'exponent' : -1}})
    sss1.register_sampler(\
        'samp_exp', 'pl', {create_updater('exponent') : create_sampler([-2, -1, 0])})
    assert 'samp_exp' in sss1

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
    for label in ['samp_exp', 'samp_freq']:
        assert label in sss2

    # Test clearing and re-registering group with list inputs
    sss2.register_group_samplers([
        ['samp_exp2', 'pl', {create_updater('exponent') : create_sampler([-2, -1, 0])}],
        ['samp_freq2', 'osc', {create_updater('freq') : create_sampler([10, 20, 30])}],
    ], clear=True)
    for old_label in ['samp_exp', 'samp_freq']:
        with raises(KeyError):
            sss2[old_label]
    for label in ['samp_exp2', 'samp_freq2']:
        assert label in sss2

def test_sim_samplers_props(tsim_samplers, tsim_params):

    # Test properties
    assert tsim_samplers.labels
    assert tsim_samplers.samplers

    # Can't directly copy object with generator - so regenerate, and test clear
    ntsim = tsim_params.copy()
    ntsamp = ntsim.to_samplers()
    ntsamp.clear(True, True, True)
    assert ntsamp.samplers == {}
    assert ntsamp.params == {}
    assert ntsamp.base == {'n_seconds': None, 'fs': None}

def test_sim_samplers_upd(tsim_samplers):

    tsim_samplers.update_sampler('samp_exp', 'n_samples', 100)
    assert tsim_samplers['samp_exp'].n_samples == 100
