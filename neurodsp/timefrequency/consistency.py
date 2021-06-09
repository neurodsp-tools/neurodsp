"""Phase consistency measures."""

from itertools import combinations

import numpy as np

###################################################################################################
###################################################################################################

def pairwise_phase_consistency(phases):
    """Compute pairwise phase consistency.

    Parameters
    ----------
    phases : 2d array
        Phases from a wavelet or Fourier analysis, in radians.

    Returns
    -------
    avg_distance : float
        Average pairwise circular distance index.
    distances : 2d array
        Pairwise circular distance indices.

    Notes
    -----

    - distance == -1: inverse phases (i.e. np.pi vs -np.pi)
    - distance ==  0: pi / 2 phase difference (i.e. np.pi vs np.pi / 2)
    - distance ==  1: equal phases (i.e. np.pi vs np.pi)

    Reference
    ---------
    Vinck, M., van Wingerden, M., Womelsdorf, T., Fries, P., & Pennartz, C. M. A. (2010).
    The pairwise phase consistency: A bias-free measure of rhythmic neuronal synchronization.
    NeuroImage, 51(1), 112–122. https://doi.org/10.1016/j.neuroimage.2010.01.073
    """

    pairs = list(combinations(np.arange(len(phases)), 2))

    distances = np.zeros((len(pairs), len(phases[0])))

    for idx, pair in enumerate(pairs):

        # Absolute angular distance
        distances[idx] = np.abs(phases[pair[0]] - phases[pair[1]]) % np.pi

        # Pairwise circular distance index (PCDI)
        distances[idx] = (np.pi - 2 * distances[idx]) / np.pi

    # Mean PCDI
    avg_distance = (2 * np.sum(distances)) / (len(phases) * (len(phases) - 1))

    return avg_distance, distances
