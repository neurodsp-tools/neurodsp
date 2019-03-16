"""Simulating time series, with periodic activity."""

from copy import deepcopy

import numpy as np
from numpy.random import rand, randn, randint
import pandas as pd
from scipy import signal

from neurodsp.sim.transients import make_osc_cycle

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
    pha_one_cycle = np.hstack([np.linspace(0, np.pi, decay_samples + 1),
                               np.linspace(-np.pi, 0, rise_samples + 1)[1:-1]])
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
    is_oscillating = _make_is_osc(n_cycles_overestimate, prob_enter_burst, prob_leave_burst)

    # Determine what the cycle properties shall be for each cycle
    periods, amps, rdsyms = _determine_cycle_properties(is_oscillating, cycle_features_use, n_tries)

    # Set up the dataframe of parameters
    df = pd.DataFrame({'is_cycle': is_oscillating, 'period': periods, 'amp': amps, 'rdsym': rdsyms})
    df['start_sample'] = np.insert(df['period'].cumsum().values[:-1], 0, 0)
    df = df[df['start_sample'] < n_samples]
    # Shorten df to only cycles that are included in the data

    # Create the signal
    sig = _sim_cycles(df)
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
            randint(low=-int(fs * jitter), high=int(fs * jitter), size=len(spk_indices))

    spks[spk_indices] = 1
    sig = np.convolve(spks, osc_cycle, 'valid')

    return sig

###################################################################################################
###################################################################################################

def _make_is_osc(n_cycles, prob_enter_burst, prob_leave_burst):
    """Create a vector describing if each cycle is oscillating."""

    is_oscillating = [None] * (n_cycles)
    is_oscillating[0] = False

    for ii in range(1, n_cycles):

        rand_num = rand()

        if is_oscillating[ii-1]:
            is_oscillating[ii] = rand_num > prob_leave_burst
        else:
            is_oscillating[ii] = rand_num  < prob_enter_burst

    return is_oscillating


def _determine_cycle_properties(is_oscillating, cycle_features, n_tries):
    """Calculate the properties for each cycle."""

    periods = np.zeros_like(is_oscillating, dtype=int)
    amps = np.zeros_like(is_oscillating, dtype=float)
    rdsyms = np.zeros_like(is_oscillating, dtype=float)

    for ind, is_osc in enumerate(is_oscillating):

        if is_osc is False:
            period = cycle_features['period_mean'] + randn() * cycle_features['period_std']
            amp = np.nan
            rdsym = np.nan

            cur_burst = {'period_mean' : np.nan, 'amp_mean' : np.nan, 'rdsym_mean' : np.nan}

        else:

            if np.isnan(cur_burst['period_mean']):

                cur_burst['period_mean'] = cycle_features['period_mean'] + randn() * cycle_features['period_burst_std']
                cur_burst['amp_mean'] = cycle_features['amp_mean'] + randn() * cycle_features['amp_burst_std']
                cur_burst['rdsym_mean'] = cycle_features['rdsym_mean'] + randn() * cycle_features['rdsym_burst_std']

            # Simulate for n_tries to get valid features -  then if any params are still negative, raise error
            for n_try in range(n_tries):

                period = cur_burst['period_mean'] + randn() * cycle_features['period_std']
                amp = cur_burst['amp_mean'] + randn() * cycle_features['amp_std']
                rdsym = cur_burst['rdsym_mean'] + randn() * cycle_features['rdsym_std']

                if period > 0 and amp > 0 and rdsym > 0 and rdsym < 1:
                    break

            # If did not break out of the for loop, no valid features were found (for/else construction)
            else:

                # Check which features are invalid - anything below 0, and rdsym above 1
                features_invalid = [label for label in ['period', 'amp', 'rdsym'] if eval(label) < 0]
                features_invalid = features_invalid + ['rdsym'] if rdsym > 1 else features_invalid

                raise ValueError("""A cycle was repeatedly simulated with invalid feature(s) for: {}
                                    (e.g. less than 0). Please change per-cycle distribution parameters
                                    (mean & std) and restart simulation.""".format(', '.join(features_invalid)))

        periods[ind] = int(period)
        amps[ind] = amp
        rdsyms[ind] = rdsym

    return periods, amps, rdsyms


def _sim_cycles(df):
    """Simulate cycle time series, given a set of parameters for each cycle."""

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

                rise_pha = np.linspace(-np.pi / 2, 0, int(row['period'] / 4))[1:]
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
                cycle_t[:decay_samples] = cycle_t[:decay_samples] * scaling + offset
                cycle_t[decay_samples:] = cycle_t[decay_samples:] * row['amp']

            else:
                cycle_t = cycle_t * row['amp']

            sig = np.append(sig, cycle_t)
            last_cycle_oscillating = True

    return sig
