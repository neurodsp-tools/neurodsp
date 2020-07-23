"""Tests for core / internal utility functions."""

from pytest import raises

from neurodsp.utils.core import *

###################################################################################################
###################################################################################################

def test_get_avg_func():

    func = get_avg_func('mean')
    assert callable(func)

    func = get_avg_func('median')
    assert callable(func)

    with raises(ValueError):
        get_avg_func('not_a_thing')
