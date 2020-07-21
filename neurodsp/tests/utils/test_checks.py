"""Tests for checker utility functions."""

from pytest import raises

from neurodsp.utils.checks import *

###################################################################################################
###################################################################################################

def test_check_param():

    # Check that valid options run without error
    check_param(0.5, 'test', [0., 1])
    check_param(0., 'test', [0., 1])
    check_param(1., 'test', [0., 1])

    # Check that invalid options raise an error
    with raises(ValueError):
        check_param(-1, 'test', [0., 1])
    with raises(ValueError):
        check_param(1.5, 'test', [0., 1])

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
