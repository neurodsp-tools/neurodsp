"""The dual threshold algorithm for detecting oscillatory bursts in a neural signal."""

import numpy as np

from neurodsp.utils.core import get_avg_func
from neurodsp.utils.decorators import multidim
from neurodsp.timefrequency.hilbert import amp_by_time

###################################################################################################
###################################################################################################

@multidim()
def detect_bursts_dual_threshold(sig, fs, dual_thresh, f_range=None,
                                 min_n_cycles=3, min_burst_duration=None,
                                 avg_type='median', magnitude_type='amplitude',
                                 **filter_kwargs):
    """Detect bursts in a signal using the dual threshold algorithm.

    Parameters
    ----------
    sig : 1d array
        Time series.
    fs : float
        Sampling rate, in Hz.
    dual_thresh : tuple of (float, float)
        Low and high threshold values for burst detection.
        Units are normalized by the average signal magnitude.
    f_range : tuple of (float, float), optional
        Frequency range, to filter signal to, before running burst detection.
        If f_range is None, then no filtering is applied prior to running burst detection.
    min_n_cycles : float, optional, default: 3
        Minimum burst duration in to keep.
        Only used if `f_range` is defined, and is used as the number of cycles at f_range[0].
    min_burst_duration : float, optional, default: None
        Minimum length of a burst, in seconds. Must be defined if not filtering.
        Only used if `f_range` is not defined, or if `min_n_cycles` is set to None.
    avg_type : {'median', 'mean'}, optional
        Averaging method to use to normalize the magnitude that is used for thresholding.
    magnitude_type : {'amplitude', 'power'}, optional
        Metric of magnitude used for thresholding.
    **filter_kwargs
        Keyword parameters to pass to `filter_signal`.

    Returns
    -------
    is_burst : 1d array
        Boolean indication of where bursts are present in the input signal.
        True indicates that a burst was detected at that sample, otherwise False.

    Examples
    --------
    Detect bursts using the dual threshold algorithm:

    >>> from neurodsp.sim import sim_combined
    >>> sig = sim_combined(n_seconds=10, fs=500,
    ...                    components={'sim_synaptic_current': {},
    ...                                'sim_bursty_oscillation' : {'freq': 10}},
    ...                    component_variances=[0.1, 0.9])
    >>> is_burst = detect_bursts_dual_threshold(sig, fs=500, dual_thresh=(1, 2), f_range=(8, 12))
    """

    if len(dual_thresh) != 2:
        raise ValueError("Invalid number of elements in 'dual_thresh' parameter")

    # Compute amplitude time series
    sig_magnitude = amp_by_time(sig, fs, f_range, remove_edges=False, **filter_kwargs)

    # Set magnitude as power or amplitude: square if power, leave as is if amplitude
    if magnitude_type not in ['amplitude', 'power']:
        raise ValueError("Invalid input for 'magnitude_type'")
    if magnitude_type == 'power':
        sig_magnitude = sig_magnitude**2

    # Calculate normalized magnitude
    sig_magnitude = sig_magnitude / get_avg_func(avg_type)(sig_magnitude)

    # Identify time periods of bursting using the 2 thresholds
    is_burst = _dual_threshold_split(sig_magnitude, dual_thresh[1], dual_thresh[0])

    # Remove bursts detected that are too short
    # Use a number of cycles defined on the frequency range, if available
    if f_range and min_n_cycles:
        min_burst_samples = int(np.ceil(min_n_cycles * fs / f_range[0]))
    # Otherwise, make sure minimum duration is set, and use that
    else:
        if min_burst_duration is None:
            raise ValueError("Minimum burst duration must be defined if not filtering "
                             "and using a number of cycles threshold.")
        min_burst_samples = int(np.ceil(min_burst_duration * fs))

    is_burst = _rmv_short_periods(is_burst, min_burst_samples)

    return is_burst.astype(bool)


def _dual_threshold_split(sig, thresh_hi, thresh_lo):
    """Identify periods that are above thresh_lo and have at least one value above thresh_hi."""

    # Find all values above thresh_hi
    # To avoid bug in later loop, do not allow first or last index to start off as 1
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
    """Remove periods that are equal to 1 for less than n_samples."""

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
    for ind in range(len(osc_starts_long)):
        is_osc[osc_starts_long[ind]:osc_ends_long[ind]] = 1

    return is_osc
