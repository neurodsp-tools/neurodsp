"""Simulating individual cycles, of different types."""

import numpy as np
from scipy.signal import gaussian, sawtooth

from neurodsp.utils.checks import check_param
from neurodsp.sim.info import get_sim_func
from neurodsp.sim.transients import sim_synaptic_kernel

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
    Simulate a half second sinusoid, corresponding to a 2 Hz cycle (frequency=1/n_seconds):

    >>> cycle = sim_cycle(n_seconds=0.5, fs=500, cycle_type='sine')

    Simulate a sawtooth cycle, corresponding to a 10 Hz cycle:

    >>> cycle = sim_cycle(n_seconds=0.1, fs=500, cycle_type='sawtooth', width=0.3)

    Notes
    -----
    Any function defined in sim.cycles as `sim_label_cycle(n_seconds, fs, **params)`,
    is accessible by this function. The `cycle_type` input must match the label.
    """

    cycle_func = get_sim_func('sim_' + cycle_type + '_cycle', modules=['cycles'])
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

    Examples
    --------
    Simulate a cycle of a 1 Hz sine wave:

    >>> cycle = sim_sine_cycle(n_seconds=1, fs=500)
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

    Examples
    --------
    Simulate a symmetric cycle of a sawtooth wave:

    >>> cycle = sim_sawtooth_cycle(n_seconds=0.25, fs=500, width=0.5)
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

    Examples
    --------
    Simulate a cycle of a gaussian wave:

    >>> cycle = sim_gaussian_cycle(n_seconds=0.2, fs=500, std=0.025)
    """

    cycle = gaussian(int(np.round(n_seconds * fs)), std * fs)

    return cycle


def sim_gaussian_asym_cycle(n_seconds, fs, beta, scale=2, shift=1):
    """Simulate an asymmetrical guassian cycle.

    Parameters
    ----------
    n_seconds : float
        Length of cycle window in seconds.
    fs : float
        Sampling frequency of the cycle simulation.
    beta : float
        Exponent controlling the amplitude asymmetry of peaks relative to the troughs.

        - `beta=0` : zeros
        - `beta=.5`: wide peaks, narrow troughs
        - `beta=1.`: symetrical peaks and troughs (sine wave)
        - `beta=5.`: wide troughs, narrow peaks

    scale : float, optional, default: 2
        Rescales the amplitude of the signal.
    shift : float, optional, default: 1
        Translate the signal along the y-axis.

    Returns
    -------
    cycle : 1d array
        Simulated asymmetrical gaussian cycle.

    Notes
    -----
    If shift or scale keyword arguments are given, default signal normalization will be bypassed.

    References
    ----------
    Lozano-Soldevilla, D., Huurne, N. T., &amp; Oostenveld, R. (2016). Neuronal Oscillations with
    Non-sinusoidal Morphology Produce Spurious Phase-to-Amplitude Coupling and Directionality.
    Frontiers in Computational Neuroscience, 10. doi:10.3389/fncom.2016.00087

    Examples
    --------
    Simulate a cycle of an asymmetrical gaussian wave:

    >>> cycle = sim_gaussian_asym_cycle(n_seconds=1, fs=500, beta=3)
    """

    check_param(beta, 'beta', [0., np.inf])

    times = create_cycle_time(n_seconds, fs)

    cycle = ((-np.cos(times) + shift) / scale)**beta

    return cycle


def sim_harmonic_asym_cycle(n_seconds, fs, phi, n_harmonics):
    """Simulate an asymmetrical cycle as a sum of harmonics.

    Parameters
    ----------
    n_seconds : float
        Length of cycle window in seconds.
    fs : float
        Sampling frequency of the cycle simulation.
    phi : float
        Phase at each harmonic.
    n_harmonics : int
        Number of harmonics to sum across.

    Returns
    -------
    cycle : 1d array
        Simulated asymmetrical harmonic cycle.

    Notes
    -----
    .. math::

      cycle = \sum_{j=1}^{j} \dfrac{1}{j^2} \cdot cos(j2\pi ft)+(j-1)*\phi

    References
    ----------
    Lozano-Soldevilla, D., Huurne, N. T., &amp; Oostenveld, R. (2016). Neuronal Oscillations with
    Non-sinusoidal Morphology Produce Spurious Phase-to-Amplitude Coupling and Directionality.
    Frontiers in Computational Neuroscience, 10. doi:10.3389/fncom.2016.00087

    Examples
    --------
    Simulate an asymmetrical cycle as the sum of harmonics:

    >>> cycle = sim_gaussian_asym_cycle(n_seconds=1, fs=500, beta=3)
    """
    times = create_cycle_time(n_seconds, fs)
    cycs = np.zeros((n_harmonics+1, len(times)))

    harmonics = np.array([freq * n_seconds for freq in range(1, n_harmonics + 2)])

    for idx, jth in enumerate(harmonics):
        cycs[idx] = (1 / jth**2) * np.cos(jth*times+(jth-1)*phi)

    cycle = np.sum(cycs, axis=0)

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

    return 2 * np.pi * 1 / n_seconds * (np.arange(fs * n_seconds) / fs)


def phase_shift_cycle(cycle, shift):
    """Phase shift a simulated cycle time series.

    Parameters
    ----------
    cycle : 1d array
        Cycle values to apply a rotation shift to.
    shift : float
        The amount to rotationally shift the cycle.
        The shift is defined as a relative proportion of cycle, between [0, 1].

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

    check_param(shift, 'shift', [0., 1.])

    shift = int(np.round(shift * len(cycle)))

    indices = range(shift, shift+len(cycle))
    cycle = cycle.take(indices, mode='wrap')

    return cycle
