"""Simulating time series, with periodic activity."""

import numpy as np

from neurodsp.utils.norm import normalize_sig
from neurodsp.utils.decorators import normalize
from neurodsp.sim.cycles import sim_cycle, phase_shift_cycle

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
    cycle : {'sine', 'asine', 'sawtooth', 'gaussian', 'exp', '2exp'}
        What type of oscillation cycle to simulate.
        See `sim_cycle` for details on cycle types and parameters.
    phase : float, optional, default: 0
        If non-zero, applies a phase shift to the oscillation by rotating the cycle.
        The shift is defined as a relative proportion of cycle, between [0, 1].
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

    # Figure out how many cycles are needed for the signal, & length of each cycle
    n_cycles = int(np.ceil(n_seconds * freq))
    n_seconds_cycle = int(np.ceil(fs / freq)) / fs

    # Create a single cycle of an oscillation
    cycle = sim_cycle(n_seconds_cycle, fs, cycle, **cycle_params)

    # Phase shift the simulated cycle
    cycle = phase_shift_cycle(cycle, phase)

    # Tile the cycle, to create the desired oscillation
    sig = np.tile(cycle, n_cycles)

    # Truncate the length of the signal to be the number of expected samples
    n_samps = int(n_seconds * fs)
    sig = sig[:n_samps]

    return sig


def sim_bursty_oscillation(n_seconds, fs, freq, enter_burst=.2, leave_burst=.2,
                           cycle='sine', **cycle_params):
    """Simulate a bursty oscillation.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    freq : float
        Oscillation frequency, in Hz.
    enter_burst : float, optional, default: 0.2
        Probability of a cycle being oscillating given the last cycle is not oscillating.
    leave_burst : float, optional, default: 0.2
        Probability of a cycle not being oscillating given the last cycle is oscillating.
    cycle : {'sine', 'asine', 'sawtooth', 'gaussian', 'exp', '2exp'}
        What type of oscillation cycle to simulate.
        See `sim_cycle` for details on cycle types and parameters.
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
    Simulate a bursty oscillation, with a low probability of bursting:

    >>> sig = sim_bursty_oscillation(n_seconds=10, fs=500, freq=5, enter_burst=0.2, leave_burst=0.8)

    Simulate a bursty oscillation, with a high probability of bursting:

    >>> sig = sim_bursty_oscillation(n_seconds=10, fs=500, freq=5, enter_burst=0.8, leave_burst=0.4)

    Simulate a bursty oscillation, of sawtooth waves:

    >>> sig = sim_bursty_oscillation(n_seconds=10, fs=500, freq=10, cycle='sawtooth', width=0.3)
    """

    # Determine number of samples & cycles
    n_samples = int(n_seconds * fs)
    n_seconds_cycle = (1/freq * fs)/fs

    # Grab normalization parameters, if any were provided
    mean = cycle_params.pop('mean', 0.)
    variance = cycle_params.pop('variance', 1.)

    # Make a single cycle of an oscillation, and normalize this cycle
    osc_cycle = sim_cycle(n_seconds_cycle, fs, cycle, **cycle_params)
    osc_cycle = normalize_sig(osc_cycle, mean, variance)

    # Calculate how many cycles are needed to tile the full signal
    n_samples_cycle = len(osc_cycle)
    n_cycles = int(np.floor(n_samples / n_samples_cycle))

    # Determine which periods will be oscillating
    is_oscillating = _make_is_osc(n_cycles, enter_burst, leave_burst)

    # Fill in the signal with cycle oscillations, for all bursting cycles
    sig = np.zeros([n_samples])
    for is_osc, cycle_ind in zip(is_oscillating, range(0, n_samples, n_samples_cycle)):
        if is_osc:
            sig[cycle_ind:cycle_ind+n_samples_cycle] = osc_cycle

    return sig

###################################################################################################
###################################################################################################

def _make_is_osc(n_cycles, enter_burst, leave_burst):
    """Create a vector describing if each cycle is oscillating, for bursting oscillations."""

    is_oscillating = [None] * (n_cycles)
    is_oscillating[0] = False

    for ii in range(1, n_cycles):

        rand_num = np.random.rand()

        if is_oscillating[ii-1]:
            is_oscillating[ii] = rand_num > leave_burst
        else:
            is_oscillating[ii] = rand_num < enter_burst

    return is_oscillating
