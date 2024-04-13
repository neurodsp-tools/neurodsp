"""Tests for neurodsp.sim.update."""

from neurodsp.sim.aperiodic import sim_powerlaw

from neurodsp.sim.update import *

###################################################################################################
###################################################################################################

def test_param_updater():

    params = {'n_seconds' : 10, 'fs' : 250, 'exponent' : None}

    upd = param_updater('exponent')
    assert callable(upd)
    upd(params, -2)
    assert params['exponent'] == -2

def test_component_updater():

    params = {'n_seconds' : 10, 'fs' : 250, 'components' : {'sim_powerlaw' : {'exponent' : None}}}

    upd = component_updater('exponent', 'sim_powerlaw')
    assert callable(upd)

    upd(params, -2)
    assert params['components']['sim_powerlaw']['exponent'] == -2

def test_create_updater():

    upd1 = create_updater('exponent')
    assert callable(upd1)

    upd2 = create_updater('exponent', 'sim_powerlaw')
    assert callable(upd2)

def test_param_iter_yielder():

    sim_params = {'n_seconds' : 5, 'fs' : 250, 'exponent' : None}
    updater = create_updater('exponent')
    values = [-2, -1, 0]

    iter_yielder = param_iter_yielder(sim_params, updater, values)
    for ind, params in enumerate(iter_yielder):
        assert isinstance(params, dict)
        for el in ['n_seconds', 'fs', 'exponent']:
            assert el in params
        assert params['exponent'] == values[ind]

def test_class_param_iter():

    sim_params = {'n_seconds' : 5, 'fs' : 250, 'exponent' : None}
    update = 'exponent'
    values = [-2, -1, 0]

    piter = ParamIter(sim_params, update, values)
    assert piter
    for ind, params in enumerate(piter):
        assert isinstance(params, dict)
        for el in ['n_seconds', 'fs', 'exponent']:
            assert el in params
        assert params['exponent'] == values[ind]

def test_param_iter():

    sim_params = {'n_seconds' : 5, 'fs' : 250, 'exponent' : None}
    update = 'exponent'
    values = [-2, -1, 0]

    # Note: no accuracy checking here (done in `test_class_param_iter`)
    piter = param_iter(sim_params, update, values)
    assert isinstance(piter, ParamIter)
    for params in piter:
        assert isinstance(params, dict)

def test_create_sampler():

    values = [-2, -1, 0]

    sampler1 = create_sampler(values, n_samples=5)
    for samp in sampler1:
        assert samp in values

    sampler2 = create_sampler(values, probs=[0.5, 0.25, 0.25], n_samples=5)
    for samp in sampler2:
        assert samp in values

def test_sample_yielder():

    # Single sampler example
    param1 = 'exponent'
    values1 = [-2, -1, 0]

    params1 = {'n_seconds' : 10, 'fs' : 250, 'exponent' : None}
    samplers1 = {
        create_updater(param1) : create_sampler(values1),
    }

    param_sampler1 = param_sample_yielder(params1, samplers1, n_samples=5)
    for ind, params in enumerate(param_sampler1):
        assert isinstance(params, dict)
        assert params['exponent'] in values1
    assert ind == 4

    # Multiple samplers example
    param2 = 'freq'
    values2 = [10, 20, 30]

    params2 = {'n_seconds' : 10, 'fs' : 250,
               'components' : {'sim_powerlaw' : {'exponent' : None},
                               'sim_oscillation' : {'freq' : None}}}
    samplers2 = {
        create_updater(param1, 'sim_powerlaw') : create_sampler(values1),
        create_updater(param2, 'sim_oscillation') : create_sampler(values2),
    }

    param_sampler2 = param_sample_yielder(params2, samplers2, n_samples=5)
    for ind, params in enumerate(param_sampler2):
        assert isinstance(params, dict)
        assert params['components']['sim_powerlaw'][param1] in values1
        assert params['components']['sim_oscillation'][param2] in values2
    assert ind == 4

def test_class_param_sampler():

    param = 'exponent'
    values = [-2, -1, 0]
    params = {'n_seconds' : 10, 'fs' : 250, 'exponent' : None}
    samplers = {create_updater(param) : create_sampler(values)}

    psampler = ParamSampler(params, samplers, n_samples=5)
    for ind, params in enumerate(psampler):
        assert isinstance(params, dict)
        assert params[param] in values
    assert ind == 4

def test_param_sampler():

    # Note: no accuracy checking here (done in `test_class_param_sampler`)
    param = 'exponent'
    values = [-2, -1, 0]
    params = {'n_seconds' : 10, 'fs' : 250, 'exponent' : None}
    samplers = {create_updater(param) : create_sampler(values)}
    psampler = param_sampler(params, samplers, n_samples=5)
    for ind, params in enumerate(psampler):
        assert isinstance(params, dict)

def test_class_sig_iter():

    params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}
    siter = SigIter(sim_powerlaw, params, n_sims=5)

    for ind, sig in enumerate(siter):
        assert isinstance(sig, np.ndarray)
    assert ind == 4

def test_sig_iter():

    # Note: no accuracy checking here (done in `test_class_sig_iter`)
    params = {'n_seconds' : 2, 'fs' : 250, 'exponent' : -1}
    siter = sig_iter(sim_powerlaw, params, n_sims=5)
    for ind, sig in enumerate(siter):
        assert isinstance(sig, np.ndarray)
    assert ind == 4
