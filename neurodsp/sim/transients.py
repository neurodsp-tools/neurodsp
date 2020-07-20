"""Simulating transients."""

from warnings import warn

import numpy as np
from scipy.signal import gaussian, sawtooth

from neurodsp.utils.data import create_times
from neurodsp.utils.checks import check_param

###################################################################################################
###################################################################################################

def sim_cycle(n_seconds, fs, cycle_type, **cycle_params):
    """Simulate a single cycle of a periodic pattern.

    Parameters
    ----------
    n_seconds : float
        Length of cycle window in seconds.
        This is NOT the period of the cycle, but the length of the returned array of the cycle.
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
        Keyword arguments for parameters of the cycle, all as float:

        * sine: None
        * asine: `rdsym`, rise-decay symmetry, from 0-1
        * sawtooth: `width`, width of the rising ramp as a proportion of the total cycle
        * gaussian: `std`, standard deviation of the gaussian kernel, in seconds
        * exp: `tau_d`, decay time, in seconds
        * 2exp: `tau_r` & `tau_d` rise time, and decay time, in seconds

    Returns
    -------
    cycle: 1d array
        Simulated cycle.

    Examples
    --------
    Simulate a half second sinusoidal cycle, corresponding to a 2 Hz cycle (frequency=1/n_seconds):

    >>> cycle = sim_cycle(n_seconds=0.5, fs=500, cycle_type='sine')

    Simulate a sawtooth cycle, corresponding to a 10 Hz cycle:

    >>> cycle = sim_cycle(n_seconds=0.1, fs=500, cycle_type='sawtooth', width=0.3)

    Notes
    -----
    Any function defined as `sim_label_cycle(n_seconds, fs, **params)`, in which label is used as
    the `cycle_type` input, is accessible by this function.
    """

    from neurodsp.sim.info import get_sim_func

    cycle_func = get_sim_func('sim_' + cycle_type + '_cycle')
    cycle = cycle_func(n_seconds, fs, **cycle_params)

    return cycle


def sim_sine_cycle(n_seconds, fs):
    """Simulate a cycle of a sine wave.

    Parameters
    ----------
    n_seconds : float
        Length of cycle window in seconds.
    fs : float
        Sampling frequency of the cycle simulation.

    Returns
    -------
    cycle : 1d array
        Simulated sine cycle.
    """

    times = create_cycle_time(n_seconds, fs)
    cycle = np.sin(times)

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
        Rise-decay symmetry of the cycle, as fraction of the period in the rise time, where:
        = 0.5 - symmetric (sine wave)
        < 0.5 - shorter rise, longer decay
        > 0.5 - longer rise, shorter decay

    Returns
    -------
    cycle : 1d array
        Simulated asymmetric cycle.

    Examples
    --------
    Simulate a 2 Hz asymmetric sine cycle:

    >>> cycle = sim_asine_cycle(n_seconds=0.5, fs=500, rdsym=0.75)
    """

    check_param(rdsym, 'rdsym', [0., 1.])

    # Determine number of samples in rise and decay periods
    n_samples = int(np.round(n_seconds * fs))
    n_rise = int(np.round(n_samples * rdsym))
    n_decay = n_samples - n_rise

    # Make phase array for an asymmetric cycle
    #   Note: the ceil & floor are so the cycle has the right number of samples if n_decay is odd
    phase = np.hstack([np.linspace(0, np.pi/2, int(np.ceil(n_rise/2)) + 1),
                       np.linspace(np.pi/2, -np.pi/2, n_decay + 1)[1:-1],
                       np.linspace(-np.pi/2, 0, int(np.floor(n_rise/2)) + 1)[:-1]])

    # Convert phase definition to signal
    cycle = np.sin(phase)

    return cycle


def sim_sawtooth_cycle(n_seconds, fs, width):
    """Simulate a cycle of a sawtooth wave.

    Parameters
    ----------
    n_seconds : float
        Length of cycle window in seconds.
    fs : float
        Sampling frequency of the cycle simulation.
    width : float
        Width of the rising ramp as a proportion of the total cycle.

    Returns
    -------
    cycle : 1d array
        Simulated sawtooth cycle.
    """

    check_param(width, 'width', [0., 1.])

    times = create_cycle_time(n_seconds, fs)
    cycle = sawtooth(times, width)

    return cycle


def sim_gaussian_cycle(n_seconds, fs, std):
    """Simulate a cycle of a gaussian.

    Parameters
    ----------
    n_seconds : float
        Length of cycle window in seconds.
    fs : float
        Sampling frequency of the cycle simulation.
    std : float
        Standard deviation of the gaussian kernel, in seconds.

    Returns
    -------
    cycle : 1d array
        Simulated gaussian cycle.
    """

    cycle = gaussian(int(np.round(n_seconds * fs)), std * fs)

    return cycle


def sim_synaptic_kernel(n_seconds, fs, tau_r, tau_d):
    """Simulate a synaptic kernel with specified time constants.

    Parameters
    ----------
    n_seconds : float
        Length of simulated kernel in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    tau_r : float
        Rise time of synaptic kernel, in seconds.
    tau_d : float
        Decay time of synaptic kernel, in seconds.

    Returns
    -------
    kernel : 1d array
        Simulated synaptic kernel.

    Notes
    -----
    Three types of kernels are available, based on combinations of time constants:

    - tau_r == tau_d      : alpha synapse
    - tau_r = 0           : instantaneous rise, with single exponential decay
    - tau_r != tau_d != 0 : double-exponential, with exponential rise and decay

    Examples
    --------
    Simulate an alpha synaptic kernel:

    >>> kernel = sim_synaptic_kernel(n_seconds=1, fs=500, tau_r=0.25, tau_d=0.25)

    Simulate a double exponential synaptic kernel:

    >>> kernel = sim_synaptic_kernel(n_seconds=1, fs=500, tau_r=0.1, tau_d=0.3)
    """

    # Create a times vector
    times = create_times(n_seconds, fs)

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


# Alias single exponential cycle from sim_synaptic kernel
def sim_exp_cycle(n_seconds, fs, tau_d):
    return sim_synaptic_kernel(n_seconds, fs, tau_r=0, tau_d=tau_d)
sim_exp_cycle.__doc__ = sim_synaptic_kernel.__doc__


# Alias double exponential cycle from sim_synaptic kernel
def sim_2exp_cycle(n_seconds, fs, tau_r, tau_d):
    return sim_synaptic_kernel(n_seconds, fs, tau_r=tau_r, tau_d=tau_d)
sim_2exp_cycle.__doc__ = sim_synaptic_kernel.__doc__


def create_cycle_time(n_seconds, fs):
    """Create a vector of time indices, in radians, for a single cycle.

    Parameters
    ----------
    n_seconds : float
        Length of simulated kernel in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.

    Returns
    -------
    1d array
        Time indices.

    Examples
    --------
    Create time indices, in radians, for a single cycle:

    >>> indices = create_cycle_time(n_seconds=1, fs=500)
    """

    return 2 * np.pi * 1 / n_seconds * (np.arange(fs * n_seconds) / fs)
