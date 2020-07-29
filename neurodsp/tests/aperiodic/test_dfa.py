"""Tests for fractal analysis using fluctuation measures."""

import pytest

from neurodsp.tests.settings import FS

from neurodsp.aperiodic.dfa import (compute_fluctuations, compute_rescaled_range,
                                    compute_detrended_fluctuation)

###################################################################################################
###################################################################################################

@pytest.mark.parametrize("method",
    ['dfa', 'rs', pytest.param(None, marks=pytest.mark.xfail(raises=ValueError))]
)
def test_compute_fluctuations(tsig, method):

    t_scales, flucs, exp = compute_fluctuations(tsig, FS, method=method)

    assert len(t_scales) == len(flucs)
    assert exp > 0


def test_compute_rescaled_range(tsig):

    rs = compute_rescaled_range(tsig, 10)

    assert isinstance(rs, float)


def test_compute_detrended_fluctuation(tsig):

    out = compute_detrended_fluctuation(tsig, 10)

    assert out >= 0
