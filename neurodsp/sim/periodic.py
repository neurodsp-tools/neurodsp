"""Simulating time series, with periodic activity."""

import numpy as np
from scipy.signal import resample

from neurodsp.utils.decorators import normalize
from neurodsp.sim.transients import sim_cycle

###################################################################################################
###################################################################################################

@normalize()
def sim_oscillation(n_seconds, fs, freq, cycle='sine', **cycle_params):
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
    **cycle_params
        Parameters for the simulated oscillation cycle.

    Returns
    -------
    sig : 1d array
        Simulated oscillation.

    Examples
    --------
    Simulate a continuous oscillation at 5 hz:

    >>> sig = sim_oscillation(n_seconds=1, fs=500, freq=5)
    """

    # Create a single cycle, that can be tiled to create the overall signal
    #osc_cycle = _make_tilable_cycle(1000, freq, cycle, **cycle_params)
    osc_cycle = _make_tilable_cycle(int(1/freq * fs), freq, cycle, **cycle_params)

    # Create the full signal by tiling the simulated single cycle the needed number of times
    n_cycles = int(np.ceil(n_seconds * freq))
    sig = np.tile(osc_cycle, n_cycles)

    # Resample cycle to desired sampling rate
    #sig = resample(sig, int(n_cycles * 1/freq * fs))

    # Truncate the length of the signal to be the number of expected samples
    #   This is done because we simulate an integer number of cycles,
    #   which may be more than the length of the requested signal
    n_samps = int(n_seconds * fs)
    sig = sig[:n_samps]

    return sig


@normalize(select_nonzero=True)
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

    >>> sig = sim_bursty_oscillation(n_seconds=10, fs=500, freq=15, enter_burst=0.8, leave_burst=0.4)

    Simulate a bursty oscillation, of sawtooth waves:

    >>> sig = sim_bursty_oscillation(n_seconds=10, fs=500, freq=10, cycle='sawtooth', width=0.3)
    """

    # Make a single cycle of an oscillation
    #   Here we add a small buffer value to the cycle, so that no values are exactly zero
    #   This is because normalization gets applied to non-zero values
    #osc_cycle = _make_tilable_cycle(1000, freq, cycle, **cycle_params) + 0.00001
    osc_cycle = _make_tilable_cycle(int(1/freq * fs), freq, cycle, **cycle_params) + 0.00001

    # Determine which periods will be oscillating
    n_cycles = int(np.floor(n_seconds * freq))
    is_oscillating = _make_is_osc(n_cycles, enter_burst, leave_burst)

    # Initialize the signal array
    n_samples_cycle = len(osc_cycle)
    n_samples = n_samples_cycle * n_cycles
    sig = np.zeros([n_samples])

    # Fill in the signal with cycle oscillations, for all bursting cycles
    for is_osc, cycle_ind in zip(is_oscillating, range(0, n_samples, n_samples_cycle)):
        if is_osc:
            sig[cycle_ind:cycle_ind+n_samples_cycle] = osc_cycle

    # Resample cycle to desired sampling rate
    #sig = resample(sig, int(n_cycles * 1/freq * fs))

    # If burst tiling isn't even with the requested signal length, pad zeros to the end
    end_points = int(n_seconds * fs) - len(sig)
    sig = np.pad(sig, (0, end_points), 'constant')

    return sig

###################################################################################################
###################################################################################################

def _make_tilable_cycle(n_samples, freq, cycle, **cycle_params):
    """Create a cycle that can be cleanly tiled.

    Parameters
    ----------
    n_samples : int
        The number of samples to create the cycle as.
    freq : float
        Oscillation frequency.
    cycle : {'sine', 'asine', 'sawtooth', 'gaussian', 'exp', '2exp'}
        What type of oscillation cycle to simulate.
        See `sim_cycle` for details on cycle types and parameters.
    **cycle_params
        Parameters for the simulated oscillation cycle.
    """

    # The cycle sampling rate is tuned to the cycle length, to help with concatenation
    n_seconds_cycle = 1 / freq
    cycle_fs = n_samples / n_seconds_cycle
    osc_cycle = sim_cycle(n_seconds_cycle, cycle_fs, cycle, **cycle_params)

    # There can be an off-by-one error in the cycle length (an extra sample than expected)
    #   This stems from an estimation artifact when creating the cycle times vector
    #   If it happens, this leads to a concatenation issue
    #   So, to avoid this we trim to the expected number of samples
    if len(osc_cycle) > n_samples:
        osc_cycle = osc_cycle[0:n_samples]

    return osc_cycle


def _make_is_osc(n_cycles, enter_burst, leave_burst):
    """Create a vector describing if each cycle is oscillating, for bursting oscillations.

    Parameters
    ----------
    n_cycles : int
        The number of cycles.
    enter_burst : float, optional, default: 0.2
        Probability of a cycle being oscillating given the last cycle is not oscillating.
    leave_burst : float, optional, default: 0.2
        Probability of a cycle not being oscillating given the last cycle is oscillating.

    Returns
    -------
    is_oscillating : 1d array
        A boolean array of whether each cycle is oscillating.
    """

    is_oscillating = [None] * (n_cycles)
    is_oscillating[0] = False

    for ii in range(1, n_cycles):

        rand_num = np.random.rand()

        if is_oscillating[ii-1]:
            is_oscillating[ii] = rand_num > leave_burst
        else:
            is_oscillating[ii] = rand_num < enter_burst

    return is_oscillating
