"""Simulating time series, with periodic activity."""

from itertools import repeat

import numpy as np

from neurodsp.utils.data import compute_nsamples, compute_cycle_nseconds, create_times
from neurodsp.utils.checks import check_param_range, check_param_options
from neurodsp.utils.decorators import normalize
from neurodsp.sim.cycles import sim_cycle, sim_normalized_cycle

###################################################################################################
###################################################################################################

@normalize
def sim_oscillation(n_seconds, fs, freq, cycle='sine', phase=0, **cycle_params):
    """Simulate an oscillation.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    freq : float
        Oscillation frequency.
    cycle : {'sine', 'asine', 'sawtooth', 'gaussian', 'exp', '2exp', 'exp_cos', 'asym_harmonic'}
            or callable
        What type of oscillation cycle to simulate.
        See `sim_cycle` for details on cycle types and parameters.
    phase : float or {'min', 'max'}, optional, default: 0
        If non-zero, applies a phase shift to the oscillation by rotating the cycle.
        If a float, the shift is defined as a relative proportion of cycle, between [0, 1].
        If 'min' or 'max', the cycle is shifted to start at it's minima or maxima.
    **cycle_params
        Parameters for the simulated oscillation cycle.

    Returns
    -------
    sig : 1d array
        Simulated oscillation.

    Examples
    --------
    Simulate a continuous sinusoidal oscillation at 5 Hz:

    >>> sig = sim_oscillation(n_seconds=1, fs=500, freq=5)

    Simulate an asymmetric oscillation at 15 Hz, with a phase shift:

    >>> sig = sim_oscillation(n_seconds=1, fs=500, freq=15,
    ...                       cycle='asine', phase=0.5, rdsym=0.75)
    """

    # Figure out how many cycles are needed for the signal
    n_cycles = int(np.ceil(n_seconds * freq))

    # Compute the number of seconds per cycle for the requested frequency
    n_seconds_cycle = compute_cycle_nseconds(freq, fs)

    # Create a single cycle of an oscillation, for the requested frequency
    cycle = sim_cycle(n_seconds_cycle, fs, cycle, phase, **cycle_params)

    # Tile the cycle, to create the desired oscillation
    sig = np.tile(cycle, n_cycles)

    # Truncate the length of the signal to be the number of expected samples
    n_samples = compute_nsamples(n_seconds, fs)
    sig = sig[:n_samples]

    return sig


def sim_bursty_oscillation(n_seconds, fs, freq, burst_def='prob', burst_params=None,
                           cycle='sine', phase=0, **cycle_params):
    """Simulate a bursty oscillation.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    freq : float
        Oscillation frequency, in Hz.
    burst_def : {'prob', 'durations'} or 1d array
        Which approach to take to define the bursts:

        - 'prob' : simulate bursts based on probabilities of entering and leaving bursts states.
        - 'durations' : simulate bursts based on lengths of bursts and inter-burst periods.
        - 1d array: use the given array as a definition of the bursts

    burst_params : dict
        Parameters for the burst definition approach.

        For the `prob` approach:

            enter_burst : float, optional, default: 0.2
                Probability of a cycle being oscillating given the last cycle is not oscillating.
            leave_burst : float, optional, default: 0.2
                Probability of a cycle not being oscillating given the last cycle is oscillating.

        For the `durations` approach:

            n_cycles_burst : int
                The number of cycles within each burst.
            n_cycles_off
                The number of non-bursting cycles, between bursts.
    cycle : {'sine', 'asine', 'sawtooth', 'gaussian', 'exp', '2exp', 'exp_cos', 'asym_harmonic'}
        What type of oscillation cycle to simulate.
        See `sim_cycle` for details on cycle types and parameters.
    phase : float or {'min', 'max'}, optional, default: 0
        If non-zero, applies a phase shift to the oscillation by rotating the cycle.
        If a float, the shift is defined as a relative proportion of cycle, between [0, 1].
        If 'min' or 'max', the cycle is shifted to start at it's minima or maxima.
    **cycle_params
        Parameters for the simulated oscillation cycle.

    Returns
    -------
    sig : 1d array
        Simulated bursty oscillation.

    Notes
    -----
    This function takes a 'tiled' approach to simulating cycles, with evenly spaced
    and consistent cycles across the whole signal, that are either oscillating or not.

    If the cycle length does not fit evenly into the simulated data length,
    then the last few samples will be non-oscillating.

    Examples
    --------
    Simulate a probabilistic bursty oscillation, with a low probability of bursting:

    >>> sig = sim_bursty_oscillation(n_seconds=10, fs=500, freq=5,
    ...                              burst_params={'enter_burst' : 0.2, 'leave_burst' : 0.8})

    Simulate a probabilistic bursty sawtooth oscillation, with a high probability of bursting:

    >>> sig = sim_bursty_oscillation(n_seconds=10, fs=500, freq=5, burst_def='prob',
    ...                              burst_params = {'enter_burst' : 0.8, 'leave_burst' : 0.4},
    ...                              cycle='sawtooth', width=0.3)

    Simulate a bursty oscillation, with specified durations:

    >>> sig = sim_bursty_oscillation(n_seconds=10, fs=500, freq=10, burst_def='durations',
    ...                              burst_params={'n_cycles_burst' : 3, 'n_cycles_off' : 3})
    """

    if isinstance(burst_def, str):
        check_param_options(burst_def, 'burst_def', ['prob', 'durations'])

    # Consistency fix: catch old parameters, and remap into burst_params
    #   This preserves the prior default values, and makes the old API work the same
    burst_params = {} if not burst_params else burst_params
    for burst_param in ['enter_burst', 'leave_burst']:
        temp = cycle_params.pop(burst_param, 0.2)
        if burst_def == 'prob' and burst_param not in burst_params:
            burst_params[burst_param] = temp

    # Simulate a normalized cycle to use for bursts
    n_seconds_cycle = compute_cycle_nseconds(freq, fs)
    osc_cycle = sim_normalized_cycle(n_seconds_cycle, fs, cycle, phase=phase, **cycle_params)

    # Calculate the number of cycles needed to tile the full signal
    n_cycles = int(np.floor(n_seconds * freq))

    # Determine which periods will be oscillating
    if isinstance(burst_def, np.ndarray):
        is_oscillating = burst_def
    elif burst_def == 'prob':
        is_oscillating = make_is_osc_prob(n_cycles, **burst_params)
    elif burst_def == 'durations':
        is_oscillating = make_is_osc_durations(n_cycles, **burst_params)

    sig = make_bursts(n_seconds, fs, is_oscillating, osc_cycle)

    return sig


def sim_variable_oscillation(n_seconds, fs, freqs, cycle='sine', phase=0, **cycle_params):
    """Simulate an oscillation that varies in frequency and cycle parameters.

    Parameters
    ----------
    n_seconds : float or None
        Simulation time, in seconds. If None, the simulation time is based on `freqs` and the
        length of `cycle_params`. If a float, the signal may be truncated or contain trailing zeros
        if not exact.
    fs : float
        Signal sampling rate, in Hz.
    freqs : float or list
        Oscillation frequencies.
    cycle : {'sine', 'asine', 'sawtooth', 'gaussian', 'exp', '2exp'} or callable
        Type of oscillation cycle to simulate.
        See `sim_cycle` for details on cycle types and parameters.
    phase : float or {'min', 'max'}, optional, default: 0
        If non-zero, applies a phase shift to the oscillation by rotating the cycle.
        If a float, the shift is defined as a relative proportion of cycle, between [0, 1].
        If 'min' or 'max', the cycle is shifted to start at it's minima or maxima.
    **cycle_params
        Parameter floats or variable lists for each cycle.

    Returns
    -------
    sig : 1d array
        Simulated bursty oscillation.

    Examples
    --------
    Simulate one second of an oscillation with a varying center frequency:

    >>> freqs = np.tile([10, 11, 10, 9], 5)
    >>> sig = sim_variable_oscillation(1, 1000, freqs)

    Simulate an oscillation with varying frequency and parameters for a given number of cycles:

    >>> freqs = [ 5, 10, 15, 20]
    >>> rdsyms = [.2, .4, .6, .8]
    >>> sig = sim_variable_oscillation(None, 1000, freqs, cycle='asine', rdsym=rdsyms)
    """

    # Ensure param lists are the same length
    param_keys = cycle_params.keys()
    param_values = list(cycle_params.values())

    param_lengths = np.array([len(params) for params in param_values
                              if isinstance(params, (tuple, list, np.ndarray))])

    # Determine the number of cycles
    if isinstance(freqs, (np.ndarray, list)):
        n_cycles = len(freqs)
    elif len(param_lengths) > 0:
        n_cycles = param_lengths[0]
    else:
        n_cycles = 1

    # Ensure freqs is iterable and an array
    freqs = np.array([freqs] * n_cycles) if isinstance(freqs, (int, float, np.number)) else freqs
    freqs = np.array(freqs) if not isinstance(freqs, np.ndarray) else freqs

    # Ensure lengths of variable params are equal
    if ~(param_lengths == len(freqs)).all():
        raise ValueError('Length of cycle_params lists and freqs must be equal.')

    # Ensure all kwargs params are iterable
    for idx, param in enumerate(param_values):
        if not isinstance(param, (tuple, list, np.ndarray)):
            param_values[idx] = [param] * n_cycles

    param_values = np.array(param_values).transpose()

    # Collect params for each cycle separately
    cycle_params = [dict(zip(param_keys, params)) for params in param_values]
    cycle_params = [{}] * len(freqs) if len(cycle_params) == 0 else cycle_params

    # Determine start/end indices, in samples
    cyc_lens = [compute_nsamples(1 / freq, fs) for freq in freqs]
    ends = np.cumsum(cyc_lens, dtype=int)
    starts = [0, *ends[:-1]]

    # Check and get the total number of signals, and initialize the output array
    n_samples = np.sum(cyc_lens) if n_seconds is None else compute_nsamples(n_seconds, fs)
    sig = np.zeros(n_samples)

    # Simulate signal, adding each cycle with spefified parameters
    for freq, params, start, end in zip(freqs, cycle_params, starts, ends):

        if start > n_samples or end > n_samples:
            break

        n_seconds_cycle = compute_cycle_nseconds(freq, fs)
        sig[start:end] = sim_normalized_cycle(n_seconds_cycle, fs, cycle, phase, **params)

    return sig


def sim_damped_oscillation(n_seconds, fs, freq, gamma, growth=None):
    """Simulate a damped relaxation oscillation.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    freq : float
        Oscillation frequency, in Hz.
    gamma : float
        Parametric dampening coefficient.
    growth : float, optional, default: None
        Logistic growth rate to smooth the heaviside step function. If None,
        a non-smoothed heaviside is used.

    Returns
    -------
    sig : 1d array
        Simulated damped relaxation oscillation.

    Notes
    -----
    - This implementation of a damped oscillation is implemented as Equation 3 of [1]_.

    References
    ----------
    .. [1] Evertz, R., Hicks, D. G., & Liley, D. T. J. (2021). Alpha blocking and 1/fβ spectral
           scaling in resting EEG can be accounted for by a sum of damped alpha band oscillatory
           processes. bioRxiv.
           DOI: https://doi.org/10.1101/2021.08.20.457060

    Examples
    --------
    Simulate a damped alpha oscillation:

    >>> sig = sim_damped_oscillation(1, 1000, 10, .1)
    """

    times = create_times(n_seconds, fs)

    exp = np.exp(-1 * gamma * times)
    cos = np.cos(2 * np.pi * freq * times)

    if growth is None:
        logit = 1
    else:
        # Smooth heaviside as a logit
        logit = 1 / (1 + np.exp(-2 * growth * times))

    return exp * cos * logit


def make_bursts(n_seconds, fs, is_oscillating, cycle):
    """Create a bursting time series by tiling when oscillations occur.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    is_oscillating : 1d array of bool
        Definition of whether each cycle is bursting or not.
    cycle : 1d array
        The cycle to use for bursts.

    Returns
    -------
    burst_sig : 1d array
        Simulated bursty oscillation.
    """

    n_samples = compute_nsamples(n_seconds, fs)
    n_samples_cycle = len(cycle)

    burst_sig = np.zeros([n_samples])
    for sig_ind, is_osc in zip(range(0, n_samples, n_samples_cycle), is_oscillating):

        # If set as an oscillating cycle, add cycle to signal
        #   The sample check is to check there are enough samples left to add a full cycle
        #   If there are not, this skipps the add, leaving zeros instead of adding part of a cycle
        if is_osc and sig_ind + n_samples_cycle < n_samples:
            burst_sig[sig_ind:sig_ind+n_samples_cycle] = cycle

    return burst_sig


def make_is_osc_prob(n_cycles, enter_burst, leave_burst):
    """Create bursting definition, based on probabilistic burst starts and stops.

    Parameters
    ----------
    n_cycles : int
        The number of cycles to simulate the burst definition for.
    enter_burst : float, optional, default: 0.2
        Probability of a cycle entering a burst, given the last cycle is not oscillating.
    leave_burst : float, optional, default: 0.2
        Probability of a cycle leaving a burst, given the last cycle is oscillating.

    Returns
    -------
    is_oscillations : 1d array of bool
        Definition of whether each cycle is bursting or not.
    """

    check_param_range(enter_burst, 'enter_burst', [0., 1.])
    check_param_range(leave_burst, 'leave_burst', [0., 1.])

    # Initialize vector of burst definitions
    is_oscillating = np.zeros(n_cycles, dtype=bool)

    for ind in range(1, n_cycles):

        rand_num = np.random.rand()

        # If prior cycle bursting, leave burst with given probability
        if is_oscillating[ind-1]:
            is_oscillating[ind] = rand_num > leave_burst
        # Otherwise, with prior cycle not bursting, enter burst with given probability
        else:
            is_oscillating[ind] = rand_num < enter_burst

    return is_oscillating


def make_is_osc_durations(n_cycles, n_cycles_burst, n_cycles_off):
    """Create bursting definition, based on cycle lengths and intervals.

    Parameters
    ----------
    n_cycles : int
        The number of cycles to simulate the burst definition for.
    n_cycles_burst : int
        Number of cycles within a burst.
    n_cycles_off : int
        Number of cycles between bursts.

    Returns
    -------
    is_oscillations : 1d array of bool
        Definition of whether each cycle is bursting or not.
    """

    # Make the burst parameters iterators
    n_cycles_burst = repeat(n_cycles_burst) if isinstance(n_cycles_burst, int) else n_cycles_burst
    n_cycles_off = repeat(n_cycles_off) if isinstance(n_cycles_off, int) else n_cycles_off

    # Initialize is oscillating
    is_oscillating = np.zeros(n_cycles, dtype=bool)

    # Fill in bursts
    ind = 0
    while ind < len(is_oscillating):

        # Within a burst, set specified cycles as bursting
        b_len = next(n_cycles_burst)
        is_oscillating[ind: ind+b_len] = True

        # Update index for the next burst
        off_len = next(n_cycles_off)
        ind = ind + b_len + off_len

    return is_oscillating


def get_burst_samples(is_oscillating, fs, freq):
    """Convert a burst definition from cycles to samples.

    Parameters
    ----------
    is_oscillating : 1d array of bool
        Definition of whether each cycle is bursting or not.
    fs : float
        Sampling rate of simulated signal, in Hz.
    freq : float
        Oscillation frequency, in Hz.

    Returns
    -------
    bursts : 1d array of bool
        Definition of whether each sample is part of a burst or not.
    """

    n_samples_cycle = compute_nsamples(1 / freq, fs)
    bursts = np.repeat(is_oscillating, n_samples_cycle)

    return bursts
