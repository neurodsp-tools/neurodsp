"""
burst.py
Analyze periods of oscillatory bursting in a neural signal
"""

from neurodsp import amp_by_time, filt, spectral
import numpy as np
import warnings
from scipy import stats, signal


def detect_bursts(x, Fs, f_range, algorithm, min_osc_periods=3,
                  dual_thresh=None,
                  deviation_type='median',
                  magnitude_type='amplitude',
                  return_amplitude=False,
                  filter_fn=None, filter_kwargs=None):
    """
    Detect periods of oscillatory bursting in a neural signal

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        The sampling rate in Hz
    f_range : (low, high), Hz
        NOTE: Not relevant in the 'bosc' method
        frequency range for narrowband signal of interest
    algorithm : string
        Name of algorithm to be used.
        'deviation' : uses multiple of amplitude in frequency range like in
                      Feingold et al., 2015 (esp. Fig. 4)
        'fixed_thresh' : uses a given threshold in the same units as 'magnitude'
                         parameter
    min_osc_periods : float
        minimum burst duration in terms of number of cycles of f_range[0]
    dual_thresh : (low, high), units depend on other parameters
        NOTE: Only used when algorithm = 'deviation' or 'fixed_thresh'
        Threshold values for determining burst
    deviation_type : string in ('median', 'mean')
        NOTE: Only used when algorithm = 'deviation' or 'fixed_thresh'
        metric to normalize magnitude used for thresholding
    magnitude_type : string in ('power', 'amplitude')
        NOTE: Only used when algorithm = 'deviation' or 'fixed_thresh'
        metric of magnitude used for thresholding
    filter_fn : filter function with required inputs (x, f_range, Fs, rmv_edge)
        NOTE: Only used when algorithm = 'deviation' or 'fixed_thresh'
        function to use to filter original time series, x
    filter_kwargs : dict
        NOTE: Only used when algorithm = 'deviation' or 'fixed_thresh'
        keyword arguments to the filter_fn
    """

    if algorithm in ['deviation', 'fixed_thresh']:

        # Set default filtering parameters
        if filter_kwargs is None:
            filter_kwargs = {}

        # Assure dual_thresh has input
        if dual_thresh is None:
            raise ValueError(
                'Need to specify dual magnitude thresholds for this algorithm')

        # Process deviation_type kwarg
        if deviation_type not in ['median', 'mean']:
            raise ValueError(
                "Invalid 'baseline' parameter. Must be 'median' or 'mean'")

        # Compute amplitude time series
        x_amplitude = amp_by_time(
            x, Fs, f_range, filter_fn=filt.filter, filter_kwargs=filter_kwargs)

        # Set magnitude as power or amplitude
        if magnitude_type == 'power':
            x_magnitude = x_amplitude**2  # np.power faster?
        elif magnitude_type == 'amplitude':
            x_magnitude = x_amplitude
        else:
            raise ValueError("Invalid 'magnitude' parameter")

        # Rescale magnitude by median or mean
        # If 'fixed_thresh', x_magnitude is unchanged
        if algorithm == 'deviation':
            # Calculate normalized magnitude
            if deviation_type == 'median':
                x_magnitude = x_magnitude / np.median(x_magnitude)
            elif deviation_type == 'mean':
                x_magnitude = x_magnitude / np.mean(x_magnitude)

        if len(dual_thresh) != 2:
            raise ValueError(
                "Invalid number of elements in 'dual_thresh' parameter")

        # Identify time periods of oscillation using the 2 thresholds
        isosc = _2threshold_split(x_magnitude, dual_thresh[1], dual_thresh[0])

    else:
        raise ValueError("Invalid 'algorithm' parameter")

    # Remove short time periods of oscillation
    min_period_length = int(np.ceil(min_osc_periods * Fs / f_range[0]))
    isosc_noshort = _rmv_short_periods(isosc, min_period_length)

    if return_amplitude:
        return isosc_noshort, x_magnitude
    else:
        return isosc_noshort


def detect_bursts_bosc(x, Fs, f_oi, f_range_slope, f_slope_excl,
                       percentile_thresh=None, plot_slope_fit=None):
    """
    Detect bursts of oscillations using the Better OSCillation detection algorithm
    described in Whitten et al., 2011, NeuroImage.

    Briefly, we estimate the background 1/f process of the signal, and use this to
    determine a power threshold at our frequency of interest.

    Rather than looking at the whole frequency band, we only look at a single frequency value.
    This frequency value is chosen as the middle of the frequency band defined,
    rounded to the nearest integer, rounded down.

    Note that the BOSC paper recommends a minimum of 3 cycles to be identified as a burst.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        The sampling rate
        The sampling rate is also used for the window size to assure that the frequency spacing is 1Hz
    f_oi : int
        frequency of the oscillation of interest (Hz)
    f_range_slope : (low, high), Hz
        frequency range over which to estimate slope
    f_slope_excl : (low, high), Hz'
        Frequency range to ignore in slope fit. This should contain the
        frequency of the oscillation of interest
    percentile_thresh : int (0 to 1)
        the probability of the chi-square distribution at which to cut off oscillation
    plot_slope_fit : bool
        whether to plot the fitted slope to the power spectrum

    Returns
    -------
    isosc : array-like 1d
        binary time series. 1 = in oscillation; 0 = not in oscillation
        if return_oscbounds is true: this is 2 lists with the start and end burst samples
    """

    # Set default threshold of chi2 distribution
    if percentile_thresh is None:
        percentile_thresh = .95

    # Set default sloep plotting
    if plot_slope_fit is None:
        plot_slope_fit = False

    # Compute Morlet Transform with 6 cycles
    f0s = np.arange(f_range_slope[0], f_range_slope[1])
    mwt = spectral.morlet_transform(x, f0s, Fs, w=6)
    mwt_power = np.abs(mwt**2)

    if sum(f_oi == f0s) != 1:
        raise ValueError('The frequency of interest, f_oi, must be within the frequency range used to fit the slope')

    # Compute average spectrum, and fit slope to it
    avg_spectrum = np.mean(mwt_power, axis=1)
    slope, offset = spectral.fit_slope(f0s, avg_spectrum, f_range_slope,
                                       fit_excl=f_slope_excl, plot_fit=plot_slope_fit)

    # Compute background power at frequency of interest, and the power
    # threshold
    f_oi_background_power = 10**(np.log10(f_oi) * slope + offset)
    power_thresh = stats.chi2.ppf(
        percentile_thresh, 2, scale=f_oi_background_power / 2)

    # Determine periods that are oscillating
    power_ts = mwt_power[f0s == f_oi][0]
    isosc = power_ts > power_thresh

    # Remove bursts less than 3 cycles in duration
    min_period_length = int(3 * Fs / f_oi)
    isosc_noshort = _rmv_short_periods(isosc, min_period_length)
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
