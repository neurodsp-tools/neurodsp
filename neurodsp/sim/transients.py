"""Simulating transients."""

from warnings import warn

import numpy as np
from scipy.signal import gaussian, sawtooth

###################################################################################################
###################################################################################################

def sim_osc_cycle(n_seconds, fs, cycle_type, **cycle_params):
    """Make one cycle of an oscillation.

    Parameters
    ----------
    n_seconds : float
        Length of cycle window in seconds.
        Note that this is NOT the period of the cycle, but the length of the returned array
        that contains the cycle, which can be (and usually is) much shorter.
    fs : float
        Sampling frequency of the cycle simulation.
    cycle_type : {'sine', 'asine', 'sawtooth', 'gaussian', 'exp', '2exp'}
        What type of cycle to simulate. Options:

        * sine: a sine wave cycle
        * asine: an asymmetric sine wave
        * sawtooth: a sawtooth wave
        * gaussian: a gaussian cycle
        * exp: a cycle with exponential decay
        * 2exp: a cycle with exponential rise and decay

    **cycle_params
        Keyword arguments for parameters of the oscillation cycle, all as float:

        * sine: None
        * asine: 'rdsym', rise-decay symmetry, from 0-1
        * sawtooth: 'width', width of the rising ramp as a proportion of the total cycle
        * gaussian: 'std', standard deviation of the gaussian kernel, in seconds
        * exp: 'tau_d', decay time, in seconds
        * 2exp: 'tau_r' & 'tau_d' rise time, and decay time, in seconds

    Returns
    -------
    cycle: 1d array
        Simulated oscillation cycle.
    """

    if cycle_type not in ['sine', 'asine', 'sawtooth', 'gaussian', 'exp', '2exp']:
        raise ValueError('Did not recognize cycle type.')

    if cycle_type == 'sine':
        cycle = np.sin(create_cycle_time(n_seconds, fs))

    elif cycle_type == 'asine':
        cycle = sim_asine_cycle(n_seconds, fs, cycle_params['rdsym'])

    elif cycle_type == 'sawtooth':
        cycle = sawtooth(create_cycle_time(n_seconds, fs), cycle_params['width'])

    elif cycle_type == 'gaussian':
        cycle = gaussian(n_seconds * fs, cycle_params['std'] * fs)

    elif cycle_type == 'exp':
        cycle = sim_synaptic_kernel(n_seconds, fs, 0, cycle_params['tau_d'])

    elif cycle_type == '2exp':
        cycle = sim_synaptic_kernel(n_seconds, fs, cycle_params['tau_r'], cycle_params['tau_d'])

    return cycle


def sim_asine_cycle(n_seconds, fs, rdsym):
    """Simulate a cycle of an asymmetric sine wave.

    Parameters
    ----------
    n_seconds : float
        Length of cycle window in seconds.
        Note that this is NOT the period of the cycle, but the length of the returned array
        that contains the cycle, which can be (and usually is) much shorter.
    fs : float
        Sampling frequency of the cycle simulation.
    rdsym : float
        Rise-decay symmetry of the oscillation, as fraction of the period in the rise time, where:
        = 0.5 - symmetric (sine wave)
        < 0.5 - shorter rise, longer decay
        > 0.5 - longer rise, shorter decay

    Returns
    -------
    cycle : 1d array
        Simulated oscillation cycle.
    """

    # Determine number of samples in rise and decay periods
    n_samples = int(n_seconds * fs)
    n_rise = int(np.round(n_samples * rdsym))
    n_decay = n_samples - n_rise

    # Make phase array for the cycle, and convert to signal
    #   Note: the ceil & floor are so the cycle has the right number of samples if n_decay is odd
    cycle = np.sin(np.hstack([np.linspace(0, np.pi/2, int(np.ceil(n_rise/2)) + 1),
                              np.linspace(np.pi/2, -np.pi/2, n_decay + 1)[1:-1],
                              np.linspace(-np.pi/2, 0, int(np.floor(n_rise/2)) + 1)[:-1]]))

    return cycle


def sim_synaptic_kernel(n_seconds, fs, tau_r, tau_d):
    """Creates synaptic kernels that with specified time constants.

    Parameters
    ----------
    n_seconds : float
        Length of simulated kernel in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz
    tau_r : float
        Rise time of synaptic kernel, in seconds.
    tau_d : float
        Decay time of synaptic kernel, in seconds.

    Returns
    -------
    kernel : 1d array
        Computed synaptic kernel with length equal to t

    Notes
    -----
    3 types of kernels are available, based on combinations of time constants:
    - tau_r == tau_d  : alpha (function) synapse
    - tau_r = 0       : instantaneous rise, (single) exponential decay
    - tau_r!=tau_d!=0 : double-exponential (rise and decay)
    """

    # NOTE: sometimes n_seconds is not exact, resulting in a slightly longer or
    #   shorter times vector, which will affect final signal length
    #   https://docs.python.org/2/tutorial/floatingpoint.html
    times = np.arange(0, n_seconds, 1./fs)

    # Kernel type: single exponential
    if tau_r == 0:
        kernel = np.exp(-times / tau_d)

    # Kernel type: alpha
    elif tau_r == tau_d:
        # I(t) = t/tau * exp(-t/tau)
        kernel = (times / tau_r) * np.exp(-times / tau_r)

    # Kernel type: double exponential
    else:
        if tau_r > tau_d:
            warn('Rise time constant should be shorter than decay time constant.')

        # I(t)=(tau_r/(tau_r-tau_d))*(exp(-t/tau_d)-exp(-t/tau_r))
        kernel = (np.exp(-times / tau_d) - np.exp(-times / tau_r))

    # Normalize the integral to 1
    kernel = kernel / np.sum(kernel)

    return kernel


def create_cycle_time(n_seconds, fs):
    """Create a vector of time indices for a single cycle.

    Parameters
    ----------
    n_seconds : float
        Length of simulated kernel in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz

    Returns
    -------
    1d array
        Time indices.
    """

    return 2*np.pi*1/n_seconds * (np.arange(fs*n_seconds)/fs)
