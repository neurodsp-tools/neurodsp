"""Analyze periods of oscillatory bursting in a neural signal."""

import numpy as np
from scipy import stats

from neurodsp import amp_by_time, filt, spectral

###################################################################################################
###################################################################################################

def detect_bursts_2threshold(x, Fs, f_range, dual_thresh,
                             min_cycles=3,
                             average_method='median',
                             magnitude_type='amplitude',
                             filter_kwargs=None):
    """Detect periods of oscillatory bursting in a neural signal.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        The sampling rate in Hz
    f_range : tuple of (float, float)
        frequency range (Hz) for narrowband signal of interest
    dual_thresh : tuple of (float, float)
        Low and high threshold values for burst detection
        Units are normalized by the average signal magnitude
    min_cycles : float
        minimum burst duration in terms of number of cycles of f_range[0]
    average_method : string in ('median', 'mean')
        NOTE: Only used when algorithm = 'deviation' or 'fixed_thresh'
        metric to normalize magnitude used for thresholding
    magnitude_type : string in ('power', 'amplitude')
        NOTE: Only used when algorithm = 'deviation' or 'fixed_thresh'
        metric of magnitude used for thresholding
    filter_kwargs : dict
        NOTE: Only used when algorithm = 'deviation' or 'fixed_thresh'
        keyword arguments to the filter_fn

    Returns
    -------
    is_burst : 1d array
        Boolean indication of where bursts are present in the input signal.
        True indicates that a burst was detected at that sample, otherwise False.
    """

    # Set default filtering parameters
    if filter_kwargs is None:
        filter_kwargs = {}

    if len(dual_thresh) != 2:
        raise ValueError(
            "Invalid number of elements in 'dual_thresh' parameter")

    # Compute amplitude time series
    x_magnitude = amp_by_time(x, Fs, f_range, filter_kwargs=filter_kwargs,
                              remove_edge_artifacts=False)

    # Set magnitude as power or amplitude
    if magnitude_type == 'power':
        x_magnitude = x_magnitude**2
    elif magnitude_type == 'amplitude':
        pass
    else:
        raise ValueError("Invalid input for 'magnitude_type'")

    # Calculate normalized magnitude
    if average_method == 'median':
        x_magnitude = x_magnitude / np.median(x_magnitude)
    elif average_method == 'mean':
        x_magnitude = x_magnitude / np.mean(x_magnitude)
    else:
        raise ValueError("Invalid input for 'average_method'")

    # Identify time periods of bursting using the 2 thresholds
    is_burst = _2threshold_split(x_magnitude, dual_thresh[1], dual_thresh[0])

    # Remove bursts detected that are too short
    min_period_length = int(np.ceil(min_cycles * Fs / f_range[0]))
    is_burst = _rmv_short_periods(is_burst, min_period_length)
    return is_burst


def get_stats(bursting, Fs):
    """Get statistics of bursts.

    Parameters
    ----------
    bursting : array-like 1d
        Boolean indication of where bursts are present in the input signal.
        Output of detect_bursts_2threshold()
    Fs : float
        The sampling rate in Hz

    Returns
    -------
    stats_dict : dict
        dict with following keys: 'N_bursts' - the number of bursts
                                  'duration_mean' - mean duration of bursts (sec)
                                  'duration_std' - std dev of burst durations (sec)
                                  'percent_burst' - % time in bursts
                                  'burst_rate' - bursts/sec
    """

    tot_time = len(bursting) / Fs

    # find burst starts and ends
    starts = np.array([])
    ends = np.array([])

    for i, index in enumerate(np.where(np.diff(bursting) != 0)[0]):
        if (i % 2) == 0:
            starts = np.append(starts, index)
        else:
            ends = np.append(ends, index)

    # duration of each burst
    durations = (ends - starts) / Fs

    stats_dict = {'N_bursts': len(starts),
                  'duration_mean': np.mean(durations),
                  'duration_std': np.std(durations),
                  'percent_burst': 100 * np.sum(bursting) / len(bursting),
                  'bursts_per_second': len(starts) / tot_time}

    return stats_dict


def _2threshold_split(x, thresh_hi, thresh_lo):
    """
    Identify periods of a time series that are above thresh_lo and have at
    least one value above thresh_hi
    """

    # Find all values above thresh_hi
    # To avoid bug in later loop, do not allow first or last index to start
    # off as 1
    x[[0, -1]] = 0
    idx_over_hi = np.where(x >= thresh_hi)[0]

    # Initialize values in identified period
    positive = np.zeros(len(x))
    positive[idx_over_hi] = 1

    # Iteratively test if a value is above thresh_lo if it is not currently in
    # an identified period
    lenx = len(x)
    for i in idx_over_hi:
        j_down = i - 1
        if positive[j_down] == 0:
            j_down_done = False
            while j_down_done is False:
                if x[j_down] >= thresh_lo:
                    positive[j_down] = 1
                    j_down -= 1
                    if j_down < 0:
                        j_down_done = True
                else:
                    j_down_done = True

        j_up = i + 1
        if positive[j_up] == 0:
            j_up_done = False
            while j_up_done is False:
                if x[j_up] >= thresh_lo:
                    positive[j_up] = 1
                    j_up += 1
                    if j_up >= lenx:
                        j_up_done = True
                else:
                    j_up_done = True

    return positive


def _rmv_short_periods(x, N):
    """Remove periods that ==1 for less than N samples"""

    if np.sum(x) == 0:
        return x

    osc_changes = np.diff(1 * x)
    osc_starts = np.where(osc_changes == 1)[0]
    osc_ends = np.where(osc_changes == -1)[0]

    if len(osc_starts) == 0:
        osc_starts = [0]
    if len(osc_ends) == 0:
        osc_ends = [len(osc_changes)]

    if osc_ends[0] < osc_starts[0]:
        osc_starts = np.insert(osc_starts, 0, 0)
    if osc_ends[-1] < osc_starts[-1]:
        osc_ends = np.append(osc_ends, len(osc_changes))

    osc_length = osc_ends - osc_starts
    osc_starts_long = osc_starts[osc_length >= N]
    osc_ends_long = osc_ends[osc_length >= N]

    is_osc = np.zeros(len(x))
    for osc in range(len(osc_starts_long)):
        is_osc[osc_starts_long[osc]:osc_ends_long[osc]] = 1
    return is_osc
