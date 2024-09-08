"""Tests for neurodsp.sim.io."""

import os
from pathlib import Path

from neurodsp.tests.settings import TEST_FILES_PATH

from neurodsp.sim.io import *

###################################################################################################
###################################################################################################

def test_fpath():

    out1 = fpath(None, 'file_name')
    assert isinstance(out1, Path)
    assert str(out1) == 'file_name'

    out2 = fpath('path', 'file_name')
    assert isinstance(out2, Path)
    assert str(out2) == 'path/file_name'

def test_save_json():

    data = {'a' : 1, 'b' : 2}
    fname = 'test_json_file.json'
    save_json(TEST_FILES_PATH / fname, data)
    assert os.path.exists(TEST_FILES_PATH / fname)

def test_save_jsonlines():

    data = [{'a' : 1, 'b' : 2}, {'a' : 10, 'b' : 20}]
    fname = 'test_jsonlines_file.json'
    save_jsonlines(TEST_FILES_PATH / fname, data)
    assert os.path.exists(TEST_FILES_PATH / fname)

def test_load_json():

    data_saved = {'a' : 1, 'b' : 2}
    fname = 'test_json_file.json'
    data_loaded = load_json(TEST_FILES_PATH / fname)
    assert data_loaded == data_saved

def test_load_jsonlines():

    data_saved = [{'a' : 1, 'b' : 2}, {'a' : 10, 'b' : 20}]
    fname = 'test_jsonlines_file.json'
    data_loaded = load_jsonlines(TEST_FILES_PATH / fname)
    assert data_loaded == data_saved

def test_save_sims_sim(tsims):

    label = 'tsims'
    folder = '_'.join([tsims.sim_func, label])

    save_sims(tsims, label, TEST_FILES_PATH)
    assert os.path.exists(TEST_FILES_PATH / folder)
    assert os.path.exists(TEST_FILES_PATH / folder / 'params.json')
    assert os.path.exists(TEST_FILES_PATH / folder / 'signals.npy')

def test_load_sims_sim(tsims):
    # Loads and tests object saved from `test_save_sims_sim`

    loaded_sims = load_sims('tsims', TEST_FILES_PATH)
    assert np.array_equal(loaded_sims.signals, tsims.signals)
    assert loaded_sims.sim_func == tsims.sim_func
    assert loaded_sims.params == tsims.params

def test_save_sims_vsim(tvsims):

    label = 'tvsims'
    folder = '_'.join([tvsims.sim_func, tvsims.update, label])

    save_sims(tvsims, label, TEST_FILES_PATH)
    assert os.path.exists(TEST_FILES_PATH / folder)
    assert os.path.exists(TEST_FILES_PATH / folder / 'params.jsonlines')
    assert os.path.exists(TEST_FILES_PATH / folder / 'signals.npy')

def test_load_sims_vsim(tvsims):
    # Loads and tests object saved from `test_save_sims_vsim`

    loaded_sims = load_sims('tvsims', TEST_FILES_PATH)
    assert np.array_equal(loaded_sims.signals, tvsims.signals)
    assert loaded_sims.sim_func == tvsims.sim_func
    assert loaded_sims.params == tvsims.params
    assert loaded_sims.update == tvsims.update
    assert loaded_sims.component == tvsims.component

def test_save_sims_msim(tmsims):

    label = 'tmsims'
    folder = '_'.join([tmsims.sim_func, tmsims.update, label])
    sub_folder = '_'.join([tmsims.sim_func, 'set'])

    save_sims(tmsims, label, TEST_FILES_PATH)
    assert os.path.exists(TEST_FILES_PATH / folder)
    for ind in range(len(tmsims)):
        assert os.path.exists(TEST_FILES_PATH / folder / (sub_folder + str(ind)))
        assert os.path.exists(TEST_FILES_PATH / folder / (sub_folder + str(ind)) / 'params.json')
        assert os.path.exists(TEST_FILES_PATH / folder / (sub_folder + str(ind)) / 'signals.npy')

def test_load_sims_msim(tmsims):
    # Loads and tests object saved from `test_save_sims_msim`

    label = 'tmsims'
    loaded_sims = load_sims(label, TEST_FILES_PATH)
    assert loaded_sims.sim_func == tmsims.sim_func
    #assert loaded_sims.params == tmsims.params
    assert loaded_sims.update == tmsims.update
    assert loaded_sims.component == tmsims.component
    for lsig, csig in zip(loaded_sims.signals, tmsims.signals):
        assert np.array_equal(lsig.signals, csig.signals)