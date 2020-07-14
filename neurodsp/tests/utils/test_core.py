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

def test_check_n_cycles():

    n_cycles = check_n_cycles(3)
    n_cycles = check_n_cycles([3, 4, 5])
    n_cycles = check_n_cycles([3, 4, 5], 3)

    with raises(ValueError):
        check_n_cycles(-1)

    with raises(ValueError):
        check_n_cycles([-1, 1])

    with raises(ValueError):
        check_n_cycles([1, 2], 3)
