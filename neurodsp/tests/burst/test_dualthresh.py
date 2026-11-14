"""Tests for neurodsp.burst.dualthresh."""

from neurodsp.tests.tsettings import FS

from neurodsp.burst.dualthresh import *
from neurodsp.burst.dualthresh import _dual_threshold_split, _rmv_short_periods

###################################################################################################
###################################################################################################

def test_detect_bursts_dual_threshold(tsig):

    # Test default settings
    bursts = detect_bursts_dual_threshold(tsig, FS, (1, 2), (8, 12))

    # Test other settings
    bursts = detect_bursts_dual_threshold(tsig, FS, (1, 2), (8, 12),
                                          avg_type='mean', magnitude_type='power')

    # Test without explicitly filtering
    bursts = detect_bursts_dual_threshold(tsig, FS, (1, 2), min_burst_duration=0.1)

def test_detect_bursts_dual_threshold_2d(tsig2d):

    bursts = detect_bursts_dual_threshold(tsig2d, FS, (1, 2), (8, 12))
    assert bursts.shape == tsig2d.shape


def test_rmv_short_periods_keeps_samples_at_true_positions():

    import numpy as np

    # A burst of 1s at indices 2-4. `_rmv_short_periods` reads burst edges off
    # `np.diff`, which marks the sample *before* each transition; failing to
    # shift those indices by one shifts every detected burst one sample early.
    sig = np.array([0, 0, 1, 1, 1, 0, 0], dtype=float)
    out = _rmv_short_periods(sig.copy(), n_samples=1)
    assert np.array_equal(out, sig)

    # A burst that runs to the final sample must keep that sample: the
    # trailing-edge sentinel has to line up with the same shifted indexing.
    sig_end = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
    out_end = _rmv_short_periods(sig_end.copy(), n_samples=3)
    expected = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
    assert np.array_equal(out_end, expected)
