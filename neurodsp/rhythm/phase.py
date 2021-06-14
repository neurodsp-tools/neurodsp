"""Phase consistency measures."""

import warnings
from importlib import import_module
from itertools import combinations, combinations_with_replacement

import numpy as np

###################################################################################################
###################################################################################################

def pairwise_phase_consistency(pha0, pha1=None, return_pairs=True, progress=None):
    """Compute pairwise phase consistency.

    Parameters
    ----------
    pha0 : 1d array
        First phases from a wavelet analysis, in radians, from -pi to pi.
    pha1 : 1d array, optional, default: None
        Second phases from a wavelet analysis, in radians, from -pi to pi.
    return_pairs : True
        Returns distance pairs as a 1d array if True.
    progress : {None, 'tqdm', 'tqdm.notebook}
        Displays tqdm progress bar.

    Returns
    -------
    distance_avg : float
        Average pairwise circular distance index.
    distances : 2d array, optional
        Pairwise circular distance indices. Only returned if ``return_pairs` is True.

    Notes
    -----

    - distance == -1: inverse phases
    - distance ==  0: pi / 2 phase difference
    - distance ==  1: equal phases


    Reference
    ---------
    Vinck, M., van Wingerden, M., Womelsdorf, T., Fries, P., & Pennartz, C. M. A. (2010).
    The pairwise phase consistency: A bias-free measure of rhythmic neuronal synchronization.
    NeuroImage, 51(1), 112–122. https://doi.org/10.1016/j.neuroimage.2010.01.073
    """

    if pha0.ndim != 1:
        raise ValueError("Phase array must be 1-dimensional.")

    if pha1 is not None and pha0.shape != pha1.shape:
        raise ValueError("Phase arrays must be the same length.")

    # Pairwise indices generator
    if pha1 is None:

        # Number of pairwise combinations
        n_combs = int((len(pha0) * (len(pha0) - 1)) / 2)

        # Exclude self-combinations (i.e. ignore (0, 0), (1, 1)...)
        iterable = enumerate(combinations(np.arange(len(pha0)), 2))

    else:

        # Include all combinations
        n_combs = int(len(pha0) ** 2)

        iterable = enumerate((row, col) for row in range(len(pha0)) for col in range(len(pha1)))

    # Initialize variables
    if return_pairs:
        cumulative = None
        distances = np.ones((len(pha0), len(pha0)))
    else:
        cumulative = 0
        distances = None

    # Optional progress bar
    if progress is not None:
        try:
            tqdm = import_module(progress)
            iterable = tqdm.tqdm(iterable, total=n_combs, dynamic_ncols=True,
                                 desc='Computing Pairwise Distances')
        except ImportError:
            pass

    # Compute distance indices
    for idx, pair in iterable:

        phi0 = pha0[pair[0]]

        if pha1 is None:
            phi1 = pha0[pair[1]]
        else:
            phi1 = pha1[pair[1]]

        # Convert range from (-pi, pi) to (0, 2pi)
        phi0 = phi0 + (2*np.pi) if phi0 < 0 else phi0
        phi1 = phi1 + (2*np.pi) if phi1 < 0 else phi1

        # Absolute angular distance
        abs_dist = np.abs(phi0 - phi1)

        # Take smaller angle (range 0 to pi)
        abs_dist = (2*np.pi) - abs_dist if abs_dist > np.pi else abs_dist

        # Pairwise circular distance index (PCDI)
        distance = (np.pi - 2 * abs_dist) / np.pi

        if isinstance(distances, np.ndarray) and pha1 is None:
            distances[pair[0], pair[1]] = distance
            distances[pair[1], pair[0]] = distance
        elif isinstance(distances, np.ndarray) and pha1 is not None:
            distances[pair[0], pair[1]] = distance
        else:
            cumulative += distance

    distance_avg = cumulative.sum() / n_combs if distances is None else np.mean(distances)

    if return_pairs:
        return distance_avg, distances
    else:
        return distance_avg
