"""Utility functions for burst analysis."""

import numpy as np

###################################################################################################
###################################################################################################

def compute_burst_stats(bursting, fs):
    """Compute statistics of bursts.

    Parameters
    ----------
    bursting : 1d array
        Boolean indication of where bursts are present in the input signal.
    fs : float
        Sampling rate, in Hz.

    Returns
    -------
    stats_dict : dict
        Contains the following keys:

        * `n_bursts`: the number of bursts
        * `duration_mean`: mean duration of bursts, in seconds
        * `duration_std`: standard deviation of burst durations, in seconds
        * `percent_burst`: percent time in bursts
        * `burst_rate`: bursts/sec

    Examples
    --------
    Compute statistics of detected bursts:

    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.burst import detect_bursts_dual_threshold

    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_synaptic_current': {},
    ...                                'sim_bursty_oscillation' : {'freq': 10}})
    >>> is_burst = detect_bursts_dual_threshold(sig, fs=500, dual_thresh=(1, 2), f_range=(8, 12))
    >>> stats_dict = compute_burst_stats(is_burst, fs=500)
    """

    tot_time = len(bursting) / fs

    change = np.diff(bursting)
    idcs, = change.nonzero()

    idcs += 1    # Get indices following the change.

    if bursting[0]:
        # If the first sample is part of a burst, prepend a zero.
        idcs = np.r_[0, idcs]

    if bursting[-1]:
        # If the last sample is part of a burst, append an index corresponding
        # to the length of signal.
        idcs = np.r_[idcs, bursting.size]

    starts = idcs[0::2]
    ends = idcs[1::2]
    durations = (ends - starts) / fs

    stats_dict = {'n_bursts': durations.size,
                  'duration_mean': durations.mean(),
                  'duration_std': durations.std(),
                  'percent_burst': 100 * sum(bursting) / len(bursting),
                  'bursts_per_second': durations.size / tot_time}

    return stats_dict
