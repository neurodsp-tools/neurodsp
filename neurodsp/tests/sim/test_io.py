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

def test_save_sims():
    pass

def test_load_sims():
    pass
