"""Tests for neurodsp.utils.core."""

from pytest import raises

from neurodsp.utils.core import *

###################################################################################################
###################################################################################################

def test_get_avg_func():

    func = get_avg_func('mean')
    assert callable(func)

    func = get_avg_func('median')
    assert callable(func)

    func = get_avg_func('sum')
    assert callable(func)

    with raises(ValueError):
        get_avg_func('not_a_thing')

def test_counter():

    c1 = counter(5)
    for ind in c1:
        pass
    assert ind == 4

    c2 = counter(None)

    for ind in c2:
        if ind == 5:
            break
    assert ind == 5
