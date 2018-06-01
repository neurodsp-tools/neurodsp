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
                      estimate_oscillating_periods_kwargs=None,
                      hilbert_increase_N=False):
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
    estimate_oscillating_periods_kwargs : dict or None
        Keyword arguments for function to find label cycles
        as in or not in an oscillation
    hilbert_increase_N : bool
        corresponding kwarg for neurodsp.amp_by_time
        If true, this zeropads the signal when computing the
        Fourier transform, which can be necessary for
        computing it in a reasonable amount of time.

    Returns
    -------
    df : pandas DataFrame
        dataframe containing several features and identifiers
        for each oscillatory cycle. Each row is one cycle.
        Note that columns are slightly different depending on if
        'center_extrema' is set to 'P' or 'T'.
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
        volt_amp : average of rise and decay voltage
        volt_peak : voltage at the peak
        volt_trough : voltage at the last trough
        volt_rdsym : voltage difference between rise and decay
        time_rdsym : fraction of cycle in the rise period
        volt_ptsym : voltage difference between peak and trough
        time_ptsym : fraction of cycle in the peak period
        oscillator_amplitude : average amplitude of the oscillation in that
                               frequency band during the cycle

    Notes
    -----
    * By default, the first extrema analyzed will be a peak,
    and the final one a trough. In order to switch the preference,
    the signal is simply inverted and columns are renamed.
    """

    # Set defaults if user input is None
    if estimate_oscillating_periods_kwargs is None:
        estimate_oscillating_periods_kwargs = {}
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
    shape_features['volt_amp'] = (shape_features['volt_decay'] + shape_features['volt_rise']) / 2

    # Comptue rise-decay symmetry features
    shape_features['volt_rdsym'] = shape_features['volt_rise'] - shape_features['volt_decay']
    shape_features['time_rdsym'] = shape_features['time_rise'] / shape_features['period']

    # Compute peak-trough symmetry features
    shape_features['volt_ptsym'] = shape_features['volt_peak'] + shape_features['volt_trough']
    shape_features['time_ptsym'] = shape_features['time_peak'] / (shape_features['time_peak'] + shape_features['time_trough'])

    # Compute average oscillatory amplitude estimate during cycle
    amp = neurodsp.amp_by_time(x, Fs, f_range, hilbert_increase_N=hilbert_increase_N)
    shape_features['oscillator_amplitude'] = [np.mean(amp[Ts[i]:Ts[i + 1]]) for i in range(N_t2t)]

    # Convert feature dictionary into a DataFrame
    df = pd.DataFrame.from_dict(shape_features)

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
        df['volt_peak'] = -df['volt_peak']
        df['volt_trough'] = -df['volt_trough']
        df['volt_rdsym'] = -df['volt_rdsym']
        df['volt_ptsym'] = -df['volt_ptsym']
        df['time_rdsym'] = 1 - df['time_rdsym']
        df['time_ptsym'] = 1 - df['time_ptsym']

    # Define whether or not each cycle is part of an oscillation
    if estimate_oscillating_periods:
        if center_extrema == 'T':
            x = -x
        df = define_true_oscillating_periods(df, x, **estimate_oscillating_periods_kwargs)

    return df


def define_true_oscillating_periods(df, x, amplitude_fraction_threshold=0,
                                    amplitude_consistency_threshold=.5,
                                    period_consistency_threshold=.5,
                                    monotonicity_threshold=.8,
                                    N_cycles_min=3):
    """
    Compute consistency between cycles and determine which are truly oscillating

    Parameters
    ----------
    df : pandas DataFrame
        dataframe of waveform features for individual cycles, trough-centered
    x : trace used to compute monotonicity
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
        1 = the same period for both cycles
        .5 = one cycle is half the duration of another cycle
    monotonicity_threshold : float (0 to 1)
        the minimum fraction of time segments between samples that must be
        going in the same direction.
        1 = rise and decay are perfectly monotonic
        .5 = both rise and decay are rising half of the time
             and decay half the time
        0 = rise period is all decaying and decay period is all rising
    N_cycles_min : int
        minimum number of cycles to be identified as truly oscillating
        needed in a row in order for them to remain identified as
        truly oscillating

    Returns
    -------
    df : pandas DataFrame
        same df as input, with additional columns for
        amp_fraction: normalized amplitude
        oscillating: True if that cycle met the criteria for an oscillatory mode

    Notes
    -----
    * The first and last period cannot be considered oscillating
    if the consistency measures are used.
    """

    # Compute normalized amplitude for all cycles
    amps = df['oscillator_amplitude'].values
    df['amp_fraction'] = (amps - np.min(amps)) / (np.max(amps) - np.min(amps))

    # Compute amplitude consistency
    C = len(df)
    amp_consists = np.ones(C) * np.nan
    rises = df['volt_rise'].values
    decays = df['volt_decay'].values
    for p in range(1, C - 1):
        consist_current = np.min([rises[p], decays[p]]) / np.max([rises[p], decays[p]])
        consist_last = np.min([rises[p - 1], decays[p]]) / np.max([rises[p - 1], decays[p]])
        consist_next = np.min([rises[p], decays[p + 1]]) / np.max([rises[p], decays[p + 1]])
        amp_consists[p] = np.min([consist_current, consist_next, consist_last])
    df['amp_consistency'] = amp_consists

    # Compute period consistency
    period_consists = np.ones(C) * np.nan
    periods = df['period'].values
    for p in range(1, C - 1):
        consist_last = np.min([periods[p], periods[p - 1]]) / np.max([periods[p], periods[p - 1]])
        consist_next = np.min([periods[p + 1], periods[p]]) / np.max([periods[p + 1], periods[p]])
        period_consists[p] = np.min([consist_next, consist_last])
    df['period_consistency'] = period_consists

    # Compute monotonicity
    monotonicity = np.ones(C) * np.nan
    if 'sample_trough' in df.columns:
        for i, row in df.iterrows():
            decay_period = x[int(row['sample_last_peak']):int(row['sample_trough'])]
            rise_period = x[int(row['sample_trough']):int(row['sample_next_peak'])]
            decay_mono = np.mean(np.diff(decay_period) < 0)
            rise_mono = np.mean(np.diff(rise_period) > 0)
            monotonicity[i] = np.mean([decay_mono, rise_mono])
    else:
        for i, row in df.iterrows():
            rise_period = x[int(row['sample_last_trough']):int(row['sample_peak'])]
            decay_period = x[int(row['sample_peak']):int(row['sample_next_trough'])]
            decay_mono = np.mean(np.diff(decay_period) < 0)
            rise_mono = np.mean(np.diff(rise_period) > 0)
            monotonicity[i] = np.mean([decay_mono, rise_mono])
    df['monotonicity'] = monotonicity

    # Compute if each period is part of an oscillation
    cycle_good_amp = df['amp_fraction'] > amplitude_fraction_threshold
    cycle_good_amp_consist = df['amp_consistency'] > amplitude_consistency_threshold
    cycle_good_period_consist = df['period_consistency'] > period_consistency_threshold
    cycle_good_monotonicity = df['monotonicity'] > monotonicity_threshold
    is_cycle = cycle_good_amp * cycle_good_amp_consist * cycle_good_period_consist * cycle_good_monotonicity
    is_cycle[0] = np.nan
    is_cycle[-1] = np.nan
    df['is_cycle'] = is_cycle
    df = _min_consecutive_cycles(df, N_cycles_min=N_cycles_min)
    df['is_cycle'] = df['is_cycle'].astype(bool)
    return df


def _min_consecutive_cycles(df_shape, N_cycles_min=3):
    '''Enforce minimum number of consecutive cycles'''
    is_cycle = np.copy(df_shape['is_cycle'].values)
    temp_cycle_count = 0
    for i, c in enumerate(is_cycle):
        if c:
            temp_cycle_count += 1
        else:
            if temp_cycle_count < N_cycles_min:
                for c_rm in range(temp_cycle_count):
                    is_cycle[i - 1 - c_rm] = False
            temp_cycle_count = 0
    df_shape['is_cycle'] = is_cycle
    return df_shape
