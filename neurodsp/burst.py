"""
burst.py
Detect periods of bursting in EEG waveforms
"""

from neurodsp import amp_by_time, filt
import numpy as np


def detect_bursts(x, Fs, f_range, algorithm, thresh, magnitudetype='amplitude',
                  return_amplitude=False, min_osc_periods=3, filter_fn=None,
                  filter_kwargs=None, **kwargs):
    """
    Detect bursts using one of several methods.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        The sampling rate in Hz
    f_range : (low, high), Hz
        frequency range for narrowband signal of interest
    algorithm : string
        Name of algorithm to be used.
        'deviation' : uses multiple of amplitude in frequency range like in
                      Feingold et al., 2015 (esp. Fig. 4)
        'fixed_thresh' : uses a given threshold in the same units as 'magnitude'
                         parameter
    thresh : (low, high), units depend on other parameters
        Threshold value(s) for determining burst
        NOTE: only one value is needed for 'slopefit'
    magnitudetype : string in ('power', 'amplitude'), optional
        metric of magnitude used for thresholding
    min_osc_periods : float, optional
        minimum length of an oscillatory period in terms of the period length of f_range[0]
    filter_fn : filter function with required inputs (x, f_range, Fs, rmv_edge)
        function to use to filter original time series, x; optional
    filter_kwargs : dict
        keyword arguments to the filter_fn
    Keyword Arguments :
        baseline : string in ('median', 'mean'), optional
            (thresh only) metric to normalize magnitude used for thresholding
    """

    # Set default filtering parameters
    if filter_kwargs is None:
        filter_kwargs = {}

    # Compute amplitude time series
    x_amplitude = amp_by_time(x, Fs, f_range, filter_fn=filt.filter, filter_kwargs=filter_kwargs)

    # Set magnitude as power or amplitude
    if magnitudetype == 'power':
        x_magnitude = x_amplitude**2  # np.power faster?
    elif magnitudetype == 'amplitude':
        x_magnitude = x_amplitude
    else:
        raise ValueError("Invalid 'magnitude' parameter")

    if algorithm in ['deviation', 'fixed_thresh']:
        if 'baseline' in kwargs:
            baseline = kwargs['baseline']

            if baseline not in ['median', 'mean']:
                raise ValueError("Invalid 'baseline' parameter. Must be 'median' or 'mean'")
        else:
            baseline = 'median'

        if algorithm == 'deviation':
            # Calculate normalized magnitude
            if baseline == 'median':
                x_magnitude = x_magnitude / np.median(x_magnitude)
            elif baseline == 'mean':
                x_magnitude = x_magnitude / np.mean(x_magnitude)
        # If 'fixed_thresh', x_magnitude is fine how it is

        if len(thresh) == 2:
            thresh_lo, thresh_hi = thresh[0], thresh[1]
        else:
            raise ValueError("Invalid number of elements in 'thresh' parameter")

    else:
        raise ValueError("Invalid 'algorithm' parameter")

    # Identify time periods of oscillation
    isosc = _2threshold_split(x_magnitude, thresh_hi, thresh_lo)

    # Remove short time periods of oscillation
    min_period_length = int(np.ceil(min_osc_periods * Fs / f_range[0]))
    isosc_noshort = _rmv_short_periods(isosc, min_period_length)

    if return_amplitude:
        return isosc_noshort, x_magnitude
    else:
        return isosc_noshort


def get_stats(bursting, Fs):
    """
    Get statistics of bursts.

    Parameters
    ----------
    bursting : array-like 1d
        binary time series, output of detect_bursts()
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

    ret_dict = {'N_bursts': len(starts),
                'duration_mean': np.mean(durations),
                'duration_std': np.std(durations),
                'percent_burst': np.sum(bursting) / len(bursting),
                'rate': len(starts) / tot_time}
    return ret_dict


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
