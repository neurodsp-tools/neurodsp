"""Simulating time series, with periodic activity."""

import warnings

import numpy as np
import pandas as pd
from scipy import signal

###################################################################################################
###################################################################################################

def sim_oscillator(n_seconds, fs, freq, rdsym=.5):
    """Simulate an oscillation.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    freq : float
        Oscillator frequency.
    rdsym : float
        Rise-decay symmetry of the oscillator, as fraction of the period in the rise time, where:
        = 0.5 - symmetric (sine wave)
        < 0.5 - shorter rise, longer decay
        > 0.5 - longer rise, shorter decay

    Returns
    -------
    osc : 1d array
        Oscillating time series.
    """

    # Compute number of samples per cycle and number of cycles
    n_samples_cycle = int(np.ceil(fs / freq))
    n_samples = int(fs * n_seconds)
    n_cycles = int(np.ceil(n_seconds * freq))

    # Determine number of samples in rise and decay periods
    rise_samples = int(np.round(n_samples_cycle * rdsym))
    decay_samples = n_samples_cycle - rise_samples

    # Make phase array for a single cycle, then repeat it
    pha_one_cycle = np.hstack([np.linspace(
        0, np.pi, decay_samples + 1), np.linspace(-np.pi, 0, rise_samples + 1)[1:-1]])
    phase_t = np.tile(pha_one_cycle, n_cycles)
    phase_t = phase_t[:n_samples]

    # Transform phase into an oscillator
    osc = np.cos(phase_t)

    return osc


def sim_bursty_oscillator(n_seconds, fs, freq, rdsym=.5, prob_enter_burst=.2,
                          prob_leave_burst=.2, cycle_features=None,
                          return_cycle_df=False, n_tries=5):
    """Simulate a bursty oscillation.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz
    freq : float
        Oscillator frequency, in Hz.
    rdsym : float
        Rise-decay symmetry of the oscillator, as fraction of the period in the rise time:

        - = 0.5: symmetric (sine wave)
        - < 0.5: shorter rise, longer decay
        - > 0.5: longer rise, shorter decay
    prob_enter_burst : float
        Probability of a cycle being oscillating given the last cycle is not oscillating.
    prob_leave_burst : float
        Probability of a cycle not being oscillating given the last cycle is oscillating.
    cycle_features : dict
        Specifies the mean and standard deviations (within and across bursts) of each cycle's
        amplitude, period, and rise-decay symmetry. This can include a complete or incomplete
        set (using defaults) of the following keys:

        * amp_mean: mean cycle amplitude
        * amp_std: standard deviation of cycle amplitude
        * amp_burst_std: standard deviation of mean amplitude for each burst
        * period_mean: mean period (computed from `freq`)
        * period_std: standard deviation of period (samples)
        * period_burst_std: standard deviation of mean period for each burst
        * rdsym_mean: mean rise-decay symmetry
        * rdsym_std: standard deviation of rdsym
        * rdsym_burst_std: standard deviation of mean rdsym for each burst
    return_cycle_df : bool
        If True, return the dataframe that contains the simulation parameters for each cycle.
        This may be useful for computing power, for example, as the power of the oscillator
        should only be considered over the times where there are bursts.
    n_tries : int, optional, default=5
        Number of times to try to resimulate cycle features when an
        invalid value is returned before raising an user error.

    Returns
    -------
    sig : 1d array
        Bursty oscillator.
    df : pd.DataFrame
        Cycle-by-cycle properties of the simulated oscillator.
    """

    # Define default parameters for cycle features
    mean_period_samples = int(fs / freq)
    cycle_features_use = {'amp_mean': 1, 'amp_burst_std': 0, 'amp_std': 0,
                          'period_mean': mean_period_samples, 'period_burst_std': 0, 'period_std': 0,
                          'rdsym_mean': rdsym, 'rdsym_burst_std': 0, 'rdsym_std': 0}

    # Overwrite default cycle features with those specified
    if cycle_features is not None:
        for k in cycle_features:
            cycle_features_use[k] = cycle_features[k]

    # Determine number of cycles to generate
    n_samples = int(n_seconds * fs)
    n_cycles_overestimate = int(np.ceil(n_samples / mean_period_samples * 2))

    # Simulate if a series of cycles are oscillating or not oscillating
    is_oscillating = [False]
    n_cycles_current = 1
    while n_cycles_current < n_cycles_overestimate:
        rand_num = np.random.rand()
        if is_oscillating[-1]:
            is_oscillating.append(rand_num > prob_leave_burst)
        else:
            is_oscillating.append(rand_num < prob_enter_burst)
        n_cycles_current += 1

    # Determine period, amp, and rdsym for each cycle
    periods = []
    amps = []
    rdsyms = []
    for is_osc in is_oscillating:
        if is_osc is False:
            period = cycle_features_use['period_mean'] + \
                np.random.randn() * cycle_features_use['period_std']
            periods.append(int(period))
            amps.append(np.nan)
            rdsyms.append(np.nan)

            current_burst_period_mean = np.nan
            current_burst_amp_mean = np.nan
            current_burst_rdsym_mean = np.nan
        else:
            if np.isnan(current_burst_period_mean):
                current_burst_period_mean = cycle_features_use['period_mean'] + \
                    np.random.randn() * cycle_features_use['period_burst_std']
                current_burst_amp_mean = cycle_features_use['amp_mean'] + \
                    np.random.randn() * cycle_features_use['amp_burst_std']
                current_burst_rdsym_mean = cycle_features_use['rdsym_mean'] + \
                    np.random.randn() * cycle_features_use['rdsym_burst_std']

            # simulate for n_tries, if any params are still negative, raise error
            tries_left = n_tries
            features_valid = False
            while features_valid is False and tries_left >= 0:
                tries_left = tries_left - 1
                period = current_burst_period_mean + \
                    np.random.randn() * cycle_features_use['period_std']
                amp = current_burst_amp_mean + \
                    np.random.randn() * cycle_features_use['amp_std']
                rdsym = current_burst_rdsym_mean + \
                    np.random.randn() * cycle_features_use['rdsym_std']

                if period > 0 and amp > 0 and rdsym > 0 and rdsym < 1:
                    features_valid = True

            if features_valid is False:
                # features are still invalid after n_tries, give up.
                features_invalid = ''
                if period <= 0:
                    features_invalid += 'period '
                if amp <= 0:
                    features_invalid += 'amp '
                if rdsym <= 0 or rdsym >= 1:
                    features_invalid += 'rdsym '
                error_str = """A cycle was repeatedly simulated with
                               invalid feature(s) for: **{:s}** (e.g. less than 0).
                               Please change per-cycle distribution parameters (mean &
                               std) and restart simulation.""".format(features_invalid)
                raise ValueError(error_str)
            else:
                periods.append(int(period))
                amps.append(amp)
                rdsyms.append(rdsym)

    df = pd.DataFrame({'is_cycle': is_oscillating, 'period': periods,
                       'amp': amps, 'rdsym': rdsyms})
    df['start_sample'] = np.insert(df['period'].cumsum().values[:-1], 0, 0)
    df = df[df['start_sample'] < n_samples]

    # Shorten df to only cycles that are included in the data

    # Simulate time series for each cycle
    sig = np.array([])
    last_cycle_oscillating = False
    for ind, row in df.iterrows():
        if row['is_cycle'] is False:
            # If last cycle was oscillating, add a decay to 0 then 0s
            if last_cycle_oscillating:
                decay_pha = np.linspace(0, np.pi / 2, int(row['period'] / 4))
                decay_t = np.cos(decay_pha) * sig[-1]
                sig = np.append(sig, decay_t)

                cycle_t = np.zeros(row['period'] - int(row['period'] / 4))
                sig = np.append(sig, cycle_t)
            else:
                # Add a blank cycle
                cycle_t = np.zeros(row['period'])
                sig = np.append(sig, cycle_t)
            last_cycle_oscillating = False
        else:
            # If last cycle was oscillating, add a decay to 0
            if not last_cycle_oscillating:
                rise_pha = np.linspace(-np.pi / 2, 0,
                                       int(row['period'] / 4))[1:]
                rise_t = np.cos(rise_pha) * row['amp']
                if len(rise_pha) > 0:
                    sig[-len(rise_t):] = rise_t

            # Add a cycle with rdsym
            rise_samples = int(np.round(row['period'] * row['rdsym']))
            decay_samples = row['period'] - rise_samples
            pha_t = np.hstack([np.linspace(0, np.pi, decay_samples + 1)[1:],
                               np.linspace(-np.pi, 0, rise_samples + 1)[1:]])
            cycle_t = np.cos(pha_t)

            # Adjust decay if the last cycle was oscillating
            if last_cycle_oscillating:
                scaling = (row['amp'] + sig[-1]) / 2
                offset = (sig[-1] - row['amp']) / 2
                cycle_t[:decay_samples] = cycle_t[:decay_samples] * \
                    scaling + offset
                cycle_t[decay_samples:] = cycle_t[decay_samples:] * row['amp']
            else:
                cycle_t = cycle_t * row['amp']
            sig = np.append(sig, cycle_t)
            last_cycle_oscillating = True
    sig = sig[:n_samples]

    if return_cycle_df:
        # Remove last row of df
        df.drop(df.index[len(df) - 1], inplace=True)
        return sig, df
    else:
        return sig


def sim_jittered_oscillator(n_seconds, fs, freq, jitter=0, cycle=('gaussian', 0.01)):
    """Simulate a jittered oscillator, as defined by the oscillator frequency,
    the oscillator cycle, and how much (in time) to jitter each period.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    freq : float
        Frequency of simulated oscillator, in Hz.
    jitter : float
        Maximum jitter of oscillation period, in seconds.
    cycle : tuple or 1d array
        Oscillation cycle used in the simulation.
        If array, it's used directly. If tuple, it is generated based on given parameters.
        Possible values:

        - ('gaussian', std): gaussian cycle, standard deviation in seconds
        - ('exp', decay time): exponential decay, decay time constant in seconds
        - ('2exp', rise time, decay time): exponential rise and decay

    Returns
    -------
    sig: 1d array
        Simulated oscillation with jitter.
    """

    # If cycle is a tuple, generate the window with given params,
    if isinstance(cycle, tuple):

        # defaults to 1 second window for a cycle, which is more than enough
        # if interested in longer period oscillations, just pass in premade cycle
        osc_cycle = make_osc_cycle(1, fs, cycle)

    # If cycle is an array, just use it to do the convolution
    else:
        osc_cycle = cycle

    # Binary "spike-train" of when each cycle should occur
    spks = np.zeros(int(n_seconds * fs + len(osc_cycle)) - 1)
    osc_period = int(fs / freq)

    # Generate oscillation "event" indices
    spk_indices = np.arange(osc_period, len(spks), osc_period)

    # Add jitter to "spike" indices
    if jitter != 0:

        spk_indices = spk_indices + \
            np.random.randint(low=-int(fs * jitter),
                              high=int(fs * jitter), size=len(spk_indices))

    spks[spk_indices] = 1
    sig = np.convolve(spks, osc_cycle, 'valid')

    return sig


def make_osc_cycle(t_ker, fs, cycle_params):
    """Make 1 cycle of oscillation.

    Parameters
    ----------
    t_ker : float
        Length of cycle window in seconds.
        Note that this is NOT the period of the cycle, but the length of the
        returned array that contains the cycle, which can be (and usually is)
        much shorter.
    fs : float
        Sampling frequency of the cycle simulation.
    cycle_params : tuple
        Defines the parameters for the oscillation cycle. Possible values:

        - ('gaussian', std): gaussian cycle, standard deviation in seconds
        - ('exp', decay time): exponential decay, decay time constant in seconds
        - ('2exp', rise time, decay time): exponential rise and decay

    Returns
    -------
    cycle: 1d array
        Simulated oscillation cycle.
    """

    if cycle_params[0] == 'gaussian':
        # cycle_params defines std in seconds
        cycle = signal.gaussian(t_ker * fs, cycle_params[1] * fs)

    elif cycle_params[0] == 'exp':
        # cycle_params defines decay time constant in seconds
        cycle = make_synaptic_kernel(t_ker, fs, 0, cycle_params[1])

    elif cycle_params[0] == '2exp':
        # cycle_params defines rise and decay time constant in seconds
        cycle = make_synaptic_kernel(
            t_ker, fs, cycle_params[1], cycle_params[2])

    else:
        raise ValueError('Did not recognize cycle type.')

    return cycle
