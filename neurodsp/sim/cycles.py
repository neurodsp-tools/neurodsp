"""Simulating individual cycles, of different types."""

import numpy as np
from scipy.signal import gaussian, sawtooth

from neurodsp.sim.info import get_sim_func
from neurodsp.utils.data import compute_nsamples
from neurodsp.utils.checks import check_param_range, check_param_options
from neurodsp.utils.decorators import normalize
from neurodsp.sim.transients import sim_synaptic_kernel

###################################################################################################
###################################################################################################

def sim_cycle(n_seconds, fs, cycle_type, phase=0, **cycle_params):
    """Simulate a single cycle of a periodic pattern.

    Parameters
    ----------
    n_seconds : float
        Length of cycle window in seconds.
        This is NOT the period of the cycle, but the length of the returned array of the cycle.
    fs : float
        Sampling frequency of the cycle simulation.
    cycle_type : {'sine', 'asine', 'sawtooth', 'gaussian', 'exp', '2exp'} or callable
        What type of cycle to simulate. Options:

        * sine: a sine wave cycle
        * asine: an asymmetric sine wave
        * sawtooth: a sawtooth wave
        * gaussian: a gaussian cycle
        * exp: a cycle with exponential decay
        * 2exp: a cycle with exponential rise and decay

    phase : float or {'min', 'max'}, optional, default: 0
        If non-zero, applies a phase shift by rotating the cycle.
        If a float, the shift is defined as a relative proportion of cycle, between [0, 1].
        If 'min' or 'max', the cycle is shifted to start at it's minima or maxima.
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
    Simulate a half second sinusoid, corresponding to a 2 Hz cycle (frequency=1/n_seconds):

    >>> cycle = sim_cycle(n_seconds=0.5, fs=500, cycle_type='sine')

    Simulate a sawtooth cycle, corresponding to a 10 Hz cycle:

    >>> cycle = sim_cycle(n_seconds=0.1, fs=500, cycle_type='sawtooth', width=0.3)

    Notes
    -----
    Any function defined in sim.cycles as `sim_label_cycle(n_seconds, fs, **params)`,
    is accessible by this function. The `cycle_type` input must match the label.
    """

    if isinstance(cycle_type, str):
        cycle_func = get_sim_func('sim_' + cycle_type + '_cycle', modules=['cycles'])
    else:
        cycle_func = cycle_type

    cycle = cycle_func(n_seconds, fs, **cycle_params)
    cycle = phase_shift_cycle(cycle, phase)

    return cycle


@normalize
def sim_normalized_cycle(n_seconds, fs, cycle_type, phase=0, **cycle_params):
    return sim_cycle(n_seconds, fs, cycle_type, phase, **cycle_params)
sim_normalized_cycle.__doc__ = sim_cycle.__doc__


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

    Examples
    --------
    Simulate a cycle of a 1 Hz sine wave:

    >>> cycle = sim_sine_cycle(n_seconds=1, fs=500)
    """

    times = create_cycle_time(n_seconds, fs)
    cycle = np.sin(times)

    return cycle


def sim_asine_cycle(n_seconds, fs, rdsym, side='both'):
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
    side : {'both', 'peak', 'trough'}
        Which side of the cycle to make asymmetric.

    Returns
    -------
    cycle : 1d array
        Simulated asymmetric cycle.

    Examples
    --------
    Simulate a 2 Hz asymmetric sine cycle:

    >>> cycle = sim_asine_cycle(n_seconds=0.5, fs=500, rdsym=0.75)
    """

    check_param_range(rdsym, 'rdsym', [0., 1.])

    # Determine number of samples
    n_samples = compute_nsamples(n_seconds, fs)
    half_sample = int(n_samples/2)

    # Calculate number of samples rising
    n_rise_tot = int(np.round(n_samples * rdsym))
    n_rise_start = int(np.ceil(n_rise_tot/2))
    n_rise_end = int(np.floor(n_rise_tot/2))

    # Calculate number of samples decaying
    n_decay_full = n_samples - n_rise_tot
    n_decay_half = half_sample - n_rise_start

    # Calculate step size of one sample, to offset ranges, as needed
    offset = (1/n_samples * 2*np.pi)

    # Create phase defintion for cycle with both extrema being asymmetric
    if side == 'both':

        phase = np.hstack([np.linspace(0, np.pi/2, n_rise_start),
                           np.linspace(np.pi/2 + offset, -np.pi/2, n_decay_full),
                           np.linspace(-np.pi/2 + offset, 0-offset, n_rise_end)])

    # Create phase defintion for cycle with only one extrema being asymmetric
    if side in ['peak', 'trough']:

        phase = np.hstack([np.linspace(0, np.pi/2, n_rise_start),
                           np.linspace(np.pi/2, np.pi, n_decay_half),
                           np.linspace(np.pi + offset, 2*np.pi-offset, half_sample)])

    # Convert phase definition to signal
    cycle = np.sin(phase)

    # For asymmetric trough, rotate and flip the generated cycle
    if side == 'trough':
        cycle = -1 * phase_shift_cycle(cycle, 0.5)

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

    Examples
    --------
    Simulate a symmetric cycle of a sawtooth wave:

    >>> cycle = sim_sawtooth_cycle(n_seconds=0.25, fs=500, width=0.5)
    """

    check_param_range(width, 'width', [0., 1.])

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

    Examples
    --------
    Simulate a cycle of a gaussian wave:

    >>> cycle = sim_gaussian_cycle(n_seconds=0.2, fs=500, std=0.025)
    """

    cycle = gaussian(compute_nsamples(n_seconds, fs), std * fs)

    return cycle


# Alias single exponential cycle from `sim_synaptic_kernel`
def sim_exp_cycle(n_seconds, fs, tau_d):
    return sim_synaptic_kernel(n_seconds, fs, tau_r=0, tau_d=tau_d)
sim_exp_cycle.__doc__ = sim_synaptic_kernel.__doc__


# Alias double exponential cycle from `sim_synaptic_kernel`
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

    return 2 * np.pi * 1 / n_seconds * (np.arange(n_seconds * fs) / fs)


def phase_shift_cycle(cycle, shift):
    """Phase shift a simulated cycle time series.

    Parameters
    ----------
    cycle : 1d array
        Cycle values to apply a rotation shift to.
    shift : float or {'min', 'max'}
        If non-zero, applies a phase shift by rotating the cycle.
        If a float, the shift is defined as a relative proportion of cycle, between [0, 1].
        If 'min' or 'max', the cycle is shifted to start at it's minima or maxima.

    Returns
    -------
    cycle : 1d array
        Rotated cycle.

    Examples
    --------
    Phase shift a simulated sine wave cycle:

    >>> cycle = sim_cycle(n_seconds=0.5, fs=500, cycle_type='sine')
    >>> shifted_cycle = phase_shift_cycle(cycle, shift=0.5)
    """

    if isinstance(shift, (float, int)):
        check_param_range(shift, 'shift', [0., 1.])
    else:
        check_param_options(shift, 'shift', ['min', 'max'])

    if shift == 'min':
        shift = np.argmin(cycle)
    elif shift == 'max':
        shift = np.argmax(cycle)
    else:
        shift = int(np.round(shift * len(cycle)))

    indices = range(shift, shift+len(cycle))
    cycle = cycle.take(indices, mode='wrap')

    return cycle
