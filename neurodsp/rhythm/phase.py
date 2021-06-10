"""Phase consistency measures."""

import warnings
from importlib import import_module
from itertools import combinations

import numpy as np

###################################################################################################
###################################################################################################

def pairwise_phase_consistency(pha0, pha1, return_pairs=True, memory_gb=2, progress=None):
    """Compute pairwise phase consistency.

    Parameters
    ----------
    pha0 : 1d array
        First phases from a wavelet analysis, in radians (i.e. lfp).
    pha1 : 1d array
        Second phases from a wavelet analysis, in radians (i.e. spikes).
    return_pairs : True
        Returns distance pairs as a 1d array if True.
    memory_gb : float, optional, default: 2
        Maximum size of the ``distances`` array, in gb. If the pairwise array is larger than this
        parameter, distances will not be stored in memory to prevent OOM error. Ignored if
        ``return_pairs`` is False.
    progress : {None, 'tqdm', 'tqdm.notebook}
        Displays tqdm progress bar.

    Returns
    -------
    distance_avg : float
        Average pairwise circular distance index.
    distances : 2d array, optional
        Pairwise circular distance indices. Only returned if ``return_pairs` is True. If
        ``memory_gb`` is less than required array size, None will be returned.

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

    if pha0.shape != pha1.shape or pha0.ndim != 1:
        raise ValueError("Phase arrays must be the same 1d length.")

    n_combs = int((len(pha0) * (len(pha0) - 1)) / 2)

    # Pairwise distance array memory limit
    gb_per_float = 8e-9
    limit_mem = n_combs * gb_per_float > memory_gb

    if limit_mem and return_pairs:
        warnings.warn("Memory limit is smaller than required distance array size. "
                      "Pairwise distances will be returned as None.")

    # Initialize variables
    if return_pairs and not limit_mem:
        cumulative = None
        distances = np.zeros(n_combs)
    else:
        cumulative = 0
        distances = None

    iterable = enumerate(combinations(np.arange(len(pha0)), 2))

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

        # Absolute angular distance
        abs_dist = abs(abs(pha0[pair[0]]) - abs(pha1[pair[1]])) % (2 * np.pi)

        # Pairwise circular distance index (PCDI)
        distance = (np.pi - 2 * abs_dist) / np.pi

        if isinstance(distances, np.ndarray):
            distances[idx] = distance
        else:
            cumulative += distance

    distance_avg = cumulative.sum() / n_combs if distances is None else np.mean(distances)

    if return_pairs:
        return distance_avg, distances
    else:
        return distance_avg
