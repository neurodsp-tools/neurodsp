"""Simulating time series, with periodic activity."""

import numpy as np

from neurodsp.utils.decorators import normalize
from neurodsp.sim.transients import sim_osc_cycle

###################################################################################################
###################################################################################################

@normalize
def sim_oscillation(n_seconds, fs, freq, cycle='sine', **cycle_params):
    """Simulate an oscillation.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    freq : float
        Oscillation frequency.
    cycle : {'sine', 'asine', 'sawtooth', 'gaussian', 'exp', '2exp'}
        What type of oscillation cycle to simulate.
        See `sim_osc_cycle` for details on cycle types and parameters.
    **cycle_params
        Parameters for the simulated oscillation cycle.

    Returns
    -------
    osc : 1d array
        Oscillating time series.
    """

    # Figure out how many cycles are needed for the signal, & length of each cycle
    n_cycles = int(np.ceil(n_seconds * freq))
    n_seconds_cycle = int(np.ceil(fs / freq)) / fs

    # Create oscillation by tiling a single cycle of the desired oscillation
    osc_cycle = sim_osc_cycle(n_seconds_cycle, fs, cycle, **cycle_params)
    osc = np.tile(osc_cycle, n_cycles)

    # Truncate the length of the signal to be the number of expected samples
    n_samps = n_seconds * fs
    osc = osc[:n_samps]

    return osc


@normalize
def sim_bursty_oscillation(n_seconds, fs, freq, enter_burst=.2, leave_burst=.2,
                           cycle='sine', **cycle_params):
    """Simulate a bursty oscillation.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz
    freq : float
        Oscillation frequency, in Hz.
    enter_burst : float
        Probability of a cycle being oscillating given the last cycle is not oscillating.
    leave_burst : float
        Probability of a cycle not being oscillating given the last cycle is oscillating.
    cycle : {'sine', 'asine', 'sawtooth', 'gaussian', 'exp', '2exp'}
        What type of oscillation cycle to simulate.
        See `sim_osc_cycle` for details on cycle types and parameters.
    **cycle_params
        Parameters for the simulated oscillation cycle.

    Returns
    -------
    sig : 1d array
        Bursty oscillation.

    Notes
    -----
    * This function takes a 'tiled' approach to simulating cycles, with evenly spaced
    and consistent cycles across the whole signal, that are either oscillating or not.
    * If the cycle length does not fit evenly into the simulated data length,
    then the last few cycle will be non-oscillating.
    """

    # Determine number of samples & cycles
    n_samples = int(n_seconds * fs)
    n_seconds_cycle = (1/freq * fs)/fs

    # Make a single cycle of an oscillation
    osc_cycle = sim_osc_cycle(n_seconds_cycle, fs, cycle, **cycle_params)
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
