"""
burst.py
Detect periods of bursting in EEG waveforms
"""

from neurodsp import amp_by_time, filt
import numpy as np

def detect(x, f_range, Fs, algorithm, thresh, magnitudetype='amplitude',
           min_osc_periods=3, filter_fn=None, filter_kwargs=None, **kwargs):
    """
    Detect bursts using one of several methods.
    
    Parameters
    ----------
    x : array-like 1d
        voltage time series
    f_range : (low, high), Hz
        frequency range for narrowband signal of interest
    Fs : float
        The sampling rate
    algorithm : string
        Name of algorithm to be used.
        'deviation' : 
    thresh : (low, high), units depend on other parameters
        Threshold value(s) for determining burst
        NOTE: only one value is needed for 'slopefit'
    magnitudetype : string in ('power', 'amplitude')
        metric of magnitude used for thresholding
    min_osc_periods : float
        minimum length of an oscillatory period in terms of the period length of f_range[0]
    filter_fn : filter function with required inputs (x, f_range, Fs, rmv_edge)
        function to use to filter original time series, x
    filter_kwargs : dict
        keyword arguments to the filter_fn
    Keyword Arguments : 
        baseline : string in ('median', 'mean')
            (thresh only) metric to normalize magnitude used for thresholding
        thresh_bandpow_pc
            (magnorm only)
        whitten parameters
    """
    
    # Set default filtering parameters
    if filter_kwargs is None:
        filter_kwargs = {}

    # Compute amplitude time series
    x_amplitude = amp_by_time(x, Fs, f_range, filter_fn=filt.filter, filter_kwargs=filter_kwargs)

    # Set magnitude as power or amplitude
    if magnitudetype == 'power':
        x_magnitude = x_amplitude**2
    elif magnitudetype == 'amplitude':
        x_magnitude = x_amplitude
    else:
        raise ValueError("Invalid 'magnitude' parameter")

    if algorithm == 'deviation':
        baseline = kwargs['baseline']
        
        # Calculate normalized magnitude
        if baseline == 'median':
            norm_mag = x_magnitude / np.median(x_magnitude)
        elif baseline == 'mean':
            norm_mag = x_magnitude / np.mean(x_magnitude)
        else:
            raise ValueError("Invalid 'baseline' parameter")

        if len(thresh) == 2:
            thresh_lo, thresh_hi = thresh[0], thresh[1]
        else:
            raise ValueError("Invalid 'baseline' parameter")
        
    else:
        raise ValueError("Invalid 'algorithm' parameter")

    # Identify time periods of oscillation
    isosc = _2threshold_split(norm_mag, thresh_hi, thresh_lo)

    # Remove short time periods of oscillation
    min_period_length = int(np.ceil(min_osc_periods * Fs / f_range[0]))
    isosc_noshort = _rmv_short_periods(isosc, min_period_length)

    return isosc_noshort


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
