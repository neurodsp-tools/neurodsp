"""Tests for neurodsp.rhythm.phase."""

from pytest import mark

import numpy as np

from neurodsp.tests.settings import FS, FREQ_SINE
from neurodsp.timefrequency import phase_by_time
from neurodsp.rhythm.phase import *

###################################################################################################
###################################################################################################

@mark.parametrize('return_pairs', [True, False])
@mark.parametrize('phase_shift', [0, .25, .5])
def test_pairwise_phase_consistency(tsig_sine, return_pairs, phase_shift):

    peaks = np.where(tsig_sine == 1.0)[0]

    pha0 = phase_by_time(tsig_sine, FS)

    # Phase shift
    sig_shift = np.roll(tsig_sine, int((FS / FREQ_SINE) * phase_shift))
    pha1 = phase_by_time(sig_shift, FS)

    # Compute consistency
    dist_avg = pairwise_phase_consistency(pha0[peaks], pha1[peaks], return_pairs, 'tqdm')

    # Unpack results if needed
    if return_pairs:

        dist_avg, dists = dist_avg[0], dist_avg[1]

        assert isinstance(dists, np.ndarray)
        assert len(dists[0]) * len(dists[1]) == len(peaks) ** 2
        assert np.mean(dists) == dist_avg

    # Expected consistency
    if phase_shift == 0:
        expected = 1
    elif phase_shift == .25:
        expected = 0
    elif phase_shift == .5:
        expected = -1

    assert isinstance(dist_avg, float)
    assert round(dist_avg) == expected

    # Test self-consistency
    dist_avg, dists = pairwise_phase_consistency(pha0[peaks], return_pairs=True)

    assert dist_avg == 1
    assert len(dists[0]) == len(dists[1]) == len(peaks)

    # Cases where arrays are invalid sizes
    try:
        pairwise_phase_consistency(pha0[peaks], pha1[peaks][:2], return_pairs, 'tqdm')
        assert False
    except ValueError:
        pass

    try:
        pairwise_phase_consistency(np.zeros((2, 2)), return_pairs, 'tqdm')
        assert False
    except ValueError:
        pass
