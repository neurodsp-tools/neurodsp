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

def test_listify():

    arg1 = 1
    out1 = listify(arg1)
    assert isinstance(out1, list)

    arg2 = [1]
    out2 = listify(arg2)
    assert isinstance(out2, list)
    assert not isinstance(out2[0], list)

    arg3 = {'a' : 1, 'b' : 2}
    out3 = listify(arg3)
    assert isinstance(out3, list)
    assert out3[0] == arg3
