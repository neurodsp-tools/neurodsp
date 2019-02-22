"""Analyze periods of oscillatory bursting in a neural signal."""

import numpy as np

from neurodsp.timefrequency import amp_by_time

###################################################################################################
###################################################################################################

def detect_bursts_dual_threshold(sig, fs, f_range, dual_thresh, min_cycles=3,
                                 average_method='median', magnitude_type='amplitude',
                                 filter_kwargs=None, verbose=True):
    """Detect periods of oscillatory bursting in a neural signal.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate, in Hz.
    f_range : tuple of (float, float)
        Frequency range, in Hz, for narrowband signal of interest.
    dual_thresh : tuple of (float, float)
        Low and high threshold values for burst detection.
        Units are normalized by the average signal magnitude.
    min_cycles : float
        Minimum burst duration in terms of number of cycles of f_range[0].
    average_method : {'median', 'mean'}, optional
        Metric to normalize magnitude used for thresholding.
    magnitude_type : {'power', 'amplitude'}, optional
        Metric of magnitude used for thresholding.
    filter_kwargs : dict, optional
        Keyword arguments to the neurodsp.filt.filter_signal().
    verbose : bool, optiona, default: True
        If True, print filter transition band and any other prints.

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
    sig_magnitude = amp_by_time(sig, fs, f_range, filter_kwargs=filter_kwargs,
                                remove_edge_artifacts=False, verbose=verbose)

    # Set magnitude as power or amplitude
    if magnitude_type == 'power':
        sig_magnitude = sig_magnitude**2
    elif magnitude_type == 'amplitude':
        pass
    else:
        raise ValueError("Invalid input for 'magnitude_type'")

    # Calculate normalized magnitude
    if average_method == 'median':
        sig_magnitude = sig_magnitude / np.median(sig_magnitude)
    elif average_method == 'mean':
        sig_magnitude = sig_magnitude / np.mean(sig_magnitude)
    else:
        raise ValueError("Invalid input for 'average_method'")

    # Identify time periods of bursting using the 2 thresholds
    is_burst = _dual_threshold_split(sig_magnitude, dual_thresh[1], dual_thresh[0])

    # Remove bursts detected that are too short
    min_period_length = int(np.ceil(min_cycles * fs / f_range[0]))
    is_burst = _rmv_short_periods(is_burst, min_period_length)

    return is_burst.astype(bool)


def compute_burst_stats(bursting, fs):
    """Get statistics of bursts.

    Parameters
    ----------
    bursting : 1d array
        Boolean indication of where bursts are present in the input signal.
        Output of detect_bursts_dualthreshold().
    fs : float
        Sampling rate, in Hz.

    Returns
    -------
    stats_dict : dict
        Contains the following keys:

        * 'N_bursts': the number of bursts
        * 'duration_mean': mean duration of bursts, in seconds
        * 'duration_std': standard deviation of burst durations, in seconds
        * 'percent_burst': percent time in bursts
        * 'burst_rate': bursts/sec
    """

    tot_time = len(bursting) / fs

    # Find burst starts and ends
    starts = np.array([])
    ends = np.array([])

    for ii, index in enumerate(np.where(np.diff(bursting) != 0)[0]):

        if (ii % 2) == 0:
            starts = np.append(starts, index)
        else:
            ends = np.append(ends, index)

    # Duration of each burst
    durations = (ends - starts) / fs

    stats_dict = {'N_bursts': len(starts),
                  'duration_mean': np.mean(durations),
                  'duration_std': np.std(durations),
                  'percent_burst': 100 * np.sum(bursting) / len(bursting),
                  'bursts_per_second': len(starts) / tot_time}

    return stats_dict


def _dual_threshold_split(sig, thresh_hi, thresh_lo):
    """Identify periods of a time series that are above thresh_lo and
    have at least one value above thresh_hi.
    """

    # Find all values above thresh_hi
    # To avoid bug in later loop, do not allow first or last index to start
    # off as 1
    sig[[0, -1]] = 0
    idx_over_hi = np.where(sig >= thresh_hi)[0]

    # Initialize values in identified period
    positive = np.zeros(len(sig))
    positive[idx_over_hi] = 1

    # Iteratively test if a value is above thresh_lo if it is not currently in an identified period
    sig_len = len(sig)

    for ind in idx_over_hi:
        j_down = ind - 1
        if positive[j_down] == 0:
            j_down_done = False
            while j_down_done is False:
                if sig[j_down] >= thresh_lo:
                    positive[j_down] = 1
                    j_down -= 1
                    if j_down < 0:
                        j_down_done = True
                else:
                    j_down_done = True

        j_up = ind + 1
        if positive[j_up] == 0:
            j_up_done = False
            while j_up_done is False:
                if sig[j_up] >= thresh_lo:
                    positive[j_up] = 1
                    j_up += 1
                    if j_up >= sig_len:
                        j_up_done = True
                else:
                    j_up_done = True

    return positive


def _rmv_short_periods(sig, n_samples):
    """Remove periods that ==1 for less than n_samples."""

    if np.sum(sig) == 0:
        return sig

    osc_changes = np.diff(1 * sig)
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
    osc_starts_long = osc_starts[osc_length >= n_samples]
    osc_ends_long = osc_ends[osc_length >= n_samples]

    is_osc = np.zeros(len(sig))
    for osc in range(len(osc_starts_long)):
        is_osc[osc_starts_long[osc]:osc_ends_long[osc]] = 1

    return is_osc
