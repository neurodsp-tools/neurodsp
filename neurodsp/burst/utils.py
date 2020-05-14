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

    starts = np.array([])
    ends = np.array([])

    for ii, index in enumerate(np.where(np.diff(bursting) != 0)[0]):

        if (ii % 2) == 0:
            starts = np.append(starts, index)
        else:
            ends = np.append(ends, index)

    durations = (ends - starts) / fs

    stats_dict = {'n_bursts': len(starts),
                  'duration_mean': np.mean(durations),
                  'duration_std': np.std(durations),
                  'percent_burst': 100 * np.sum(bursting) / len(bursting),
                  'bursts_per_second': len(starts) / tot_time}

    return stats_dict
