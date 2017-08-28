"""
cycles.py
Quantify the shape of oscillatory waveforms on a cycle-by-cycle basis
"""

import numpy as np
import pandas as pd
import neurodsp
import warnings


def features_by_cycle(x, Fs, f_range, center_extrema='P',
                      find_extrema_kwargs=None,
                      estimate_oscillating_periods=False,
                      true_oscillating_periods_kwargs=None):
    """
    Calculate several features of an oscillation's waveform
    shape for each cycle in a recording.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        sampling rate (Hz)
    f_range : (low, high), Hz
        frequency range for narrowband signal of interest
    center_extrema : str
        The center extrema in the cycle
        'P' : cycles are defined trough-to-trough
        'T' : cycles are defined peak-to-peak
    find_extrema_kwargs : dict or None
        Keyword arguments for function to find peaks and
        troughs (find_extrema) to change filter
        parameters or boundary
    estimate_oscillating_periods: bool
        if True, call _define_true_oscillating_periods to
        declare each cycle as in an oscillating or not.
    true_oscillating_periods_kwargs : dict or None
        Keyword arguments for function to find label cycles
        as in or not in an oscillation

    Returns
    -------
    df : pandas DataFrame
        dataframe containing several features and identifiers
        for each oscillatory cycle. Each row is one cycle.
        Each column is described below for peak-centered cycles,
        but are similar for trough-centered cycles:
        sample_peak : sample of 'x' at which the peak occurs
        sample_zerox_decay : sample of the decaying zerocrossing
        sample_zerox_rise : sample of the rising zerocrossing
        sample_last_trough : sample of the last trough
        sample_next_trough : sample of the next trough
        period : period of the cycle
        time_decay : time between peak and next trough
        time_rise : time between peak and previous trough
        time_peak : time between rise and decay zerocrosses
        time_trough : duration of previous trough estimated by zerocrossings
        volt_decay : voltage change between peak and next trough
        volt_rise : voltage change between peak and previous trough
        volt_peak : voltage at the peak
        volt_trough : voltage at the last trough
        volt_rdsym : voltage difference between rise and decay
        time_rdsym : fraction of cycle in the rise period
        volt_ptsym : voltage difference between peak and trough
        time_ptsym : fraction of cycle in the peak period
        oscillator_amplitude : average amplitude of the oscillation in that frequency band during the cycle

    Notes
    -----
    * By default, the first extrema analyzed will be a peak,
    and the final one a trough. In order to switch the preference,
    the signal is simply inverted and columns are renamed.
    """

    # Set defaults if user input is None
    if true_oscillating_periods_kwargs is None:
        true_oscillating_periods_kwargs = {}
    if find_extrema_kwargs is None:
        find_extrema_kwargs = {}
    else:
        # Raise warning if switch from peak start to trough start
        if 'first_extrema' in find_extrema_kwargs.keys():
            raise ValueError('This function has been designed to assume that\
                              the first extrema identified will be a peak.\
                              This cannot be overwritten.')

    # Negate signal if to analyze trough-centered cycles
    if center_extrema == 'P':
        pass
    elif center_extrema == 'T':
        x = -x
    else:
        raise ValueError('Parameter "center_extrema" must be either "P" or "T"')

    # Find peak and trough locations in the signal
    Ps, Ts = neurodsp.shape.find_extrema(x, Fs, f_range, **find_extrema_kwargs)

    # Find zero-crossings
    zeroxR, zeroxD = neurodsp.shape.find_zerox(x, Ps, Ts)

    # Determine number of cycles
    N_t2t = len(Ts) - 1

    # For each cycle, identify the sample of each extrema and zerocrossing
    shape_features = {}
    shape_features['sample_peak'] = Ps[1:]
    shape_features['sample_zerox_decay'] = zeroxD[1:]
    shape_features['sample_zerox_rise'] = zeroxR
    shape_features['sample_last_trough'] = Ts[:-1]
    shape_features['sample_next_trough'] = Ts[1:]

    # Compute duration of period
    shape_features['period'] = shape_features['sample_next_trough'] - \
        shape_features['sample_last_trough']

    # Compute duration of peak
    half_decay_time = (zeroxD[1:] - Ps[1:])
    half_rise_time = (Ps[1:] - zeroxR)
    shape_features['time_peak'] = half_decay_time + half_rise_time

    # Compute duration of last trough
    half_decay_time = (Ts[:-1] - zeroxD[:-1])
    half_rise_time = (zeroxR - Ts[:-1])
    shape_features['time_trough'] = half_decay_time + half_rise_time

    # Determine extrema voltage
    shape_features['volt_peak'] = x[Ps[1:]]
    shape_features['volt_trough'] = x[Ts[:-1]]

    # Determine rise and decay characteristics
    shape_features['time_decay'] = (Ts[1:] - Ps[1:])
    shape_features['time_rise'] = (Ps[1:] - Ts[:-1])

    shape_features['volt_decay'] = x[Ps[1:]] - x[Ts[1:]]
    shape_features['volt_rise'] = x[Ps[1:]] - x[Ts[:-1]]

    # Comptue rise-decay symmetry features
    shape_features['volt_rdsym'] = shape_features['volt_rise'] - shape_features['volt_decay']
    shape_features['time_rdsym'] = shape_features['time_rise'] / shape_features['period']

    # Compute peak-trough symmetry features
    shape_features['volt_ptsym'] = shape_features['volt_peak'] - shape_features['volt_trough']
    shape_features['time_ptsym'] = shape_features['time_peak'] / (shape_features['time_peak'] + shape_features['time_trough'])

    # Compute average oscillatory amplitude estimate during cycle
    amp = neurodsp.amp_by_time(x, Fs, f_range)
    shape_features['oscillator_amplitude'] = [np.mean(amp[Ts[i]:Ts[i+1]]) for i in range(N_t2t)]

    # Convert feature dictionary into a DataFrame
    df = pd.DataFrame.from_dict(shape_features)

    # Define whether or not each cycle is part of an oscillation
    if estimate_oscillating_periods:
        df = _define_true_oscillating_periods(df, **true_oscillating_periods_kwargs)

    # Rename columns if they are actually trough-centered
    if center_extrema == 'T':
        rename_dict = {'sample_peak': 'sample_trough',
                       'sample_zerox_decay': 'sample_zerox_rise',
                       'sample_zerox_rise': 'sample_zerox_decay',
                       'sample_last_trough': 'sample_last_peak',
                       'sample_next_trough': 'sample_next_peak',
                       'time_peak': 'time_trough',
                       'time_trough': 'time_peak',
                       'volt_peak': 'volt_trough',
                       'volt_trough': 'volt_peak',
                       'time_rise': 'time_decay',
                       'time_decay': 'time_rise',
                       'volt_rise': 'volt_decay',
                       'volt_decay': 'volt_rise'}
        df.rename(columns=rename_dict, inplace=True)

        # Need to reverse symmetry measures
        df['volt_rdsym'] = -df['volt_rdsym']
        df['volt_ptsym'] = -df['volt_ptsym']
        df['time_rdsym'] = 1 - df['time_rdsym']
        df['time_ptsym'] = 1 - df['time_ptsym']

    return df


def _define_true_oscillating_periods(df, restrict_by_amplitude=True,
                                     restrict_by_amplitude_consistency=True,
                                     restrict_by_period_consistency=True,
                                     amplitude_fraction_threshold=.5,
                                     amplitude_consistency_threshold=.5,
                                     period_consistency_threshold=.5):
    """
    Denote which cycles in df meet the criteria to be in an oscillatory mode

    Parameters
    ----------
    df : pandas DataFrame
        dataframe of waveform features for individual cyclces
    restrict_by_amplitude : bool
        if True, require a cycle to have an amplitude above the threshold
        set in 'amplitude_fraction_threshold' in order to be considered
        in an oscillatory mode
    restrict_by_amplitude_consistency : bool
        if True, require the rise and decays within the cycle of interest
        and the two adjacent cycles to have consistent magnitudes,
        as defined by the 'amplitude_consistency_threshold' parameter
    restrict_by_period_consistency : bool
        if True, require an extrema to be adjacent to 2 cycles with consistent
        periods in order to be considered to be in an oscillatory mode
    amplitude_fraction_threshold : float (0 to 1)
        the minimum normalized amplitude a cycle must have
        in order to be considered in an oscillation.
        0 = the minimum amplitude across all cycles
        .5 = the median amplitude across all cycles
        1 = the maximum amplitude across all cycles
    amplitude_consistency_threshold : float (0 to 1)
        the minimum normalized difference in rise and decay magnitude
        to be considered as in an oscillatory mode
        1 = the same amplitude for the rise and decay
        .5 = the rise (or decay) is half the amplitude of the decay (rise)
    period_consistency_threshold : float (0 to 1)
        the minimum normalized difference in period between two adjacent cycles
        to be considered as in an oscillatory mode
        1 corresponds to the same period for both cycles
        .5 corresponds to one cycle being half the duration of another cycle

    Returns
    -------
    df : pandas DataFrame
        same df as input, except now with an extra boolean column,
        oscillating: True if that cycle met the criteria for an oscillatory mode

    Notes
    -----
    * The first and last period cannot be considered oscillating
    if the consistency measures are used.
    """

    # Make a binary array to indicate if a peak is in an oscillatory mode ("good")
    P = len(df)
    cycle_good_amp = np.zeros(P, dtype=bool)
    cycle_good_amp_consist = np.zeros(P, dtype=bool)
    cycle_good_period_consist = np.zeros(P, dtype=bool)

    # Compute if each cycle meets the amplitude reqt
    if restrict_by_amplitude:
        amps = df['oscillator_amplitude'].values
        ampnorm = (amps - np.min(amps)) / (np.max(amps)-np.min(amps))
        cycle_good_amp = ampnorm > amplitude_fraction_threshold
    else:
        cycle_good_amp = np.ones(P, dtype=bool)

    # Compute if each cycle meets the amplitude consistency reqt
    if restrict_by_amplitude_consistency:
        rises = df['volt_rise'].values
        decays = df['volt_decay'].values
        for p in range(1, P - 1):
            if cycle_good_amp[p]:
                frac1 = np.min([rises[p], decays[p]]) / np.max([rises[p], decays[p]])
                frac2 = np.min([rises[p], decays[p - 1]]) / \
                    np.max([rises[p], decays[p - 1]])
                frac3 = np.min([rises[p + 1], decays[p]]) / \
                    np.max([rises[p + 1], decays[p]])
                if np.min([frac1, frac2, frac3]) >= amplitude_consistency_threshold:
                    cycle_good_amp_consist[p] = True
    else:
        cycle_good_amp_consist = np.ones(P, dtype=bool)

    # Compute if each cycle meets the period consistency reqt
    if restrict_by_period_consistency:
        P_times = df['sample_peak'].values
        for p in range(1, P - 1):
            if np.logical_and(cycle_good_amp[p], cycle_good_amp_consist[p]):
                p1 = P_times[p] - P_times[p - 1]
                p2 = P_times[p + 1] - P_times[p]
                frac = np.min([p1, p2]) / np.max([p1, p2])
                if frac > period_consistency_threshold:
                    cycle_good_period_consist[p] = True
    else:
        cycle_good_period_consist = np.ones(P, dtype=bool)

    # Add indication for oscillating cycles to dataframe
    cycle_good = np.logical_and(cycle_good_amp, cycle_good_amp_consist)
    cycle_good = np.logical_and(cycle_good, cycle_good_period_consist)

    # Make first and last cycle nan if consistency measure used
    if restrict_by_amplitude_consistency or restrict_by_period_consistency:
        cycle_good[0] = np.nan
        cycle_good[-1] = np.nan

    # Add oscillating detector to the dataframe
    df['oscillating'] = cycle_good
    return df
