"""Simulating time series, with transients."""

import warnings

import numpy as np
from scipy import signal

###################################################################################################
###################################################################################################

def make_osc_cycle(t_ker, fs, cycle_params):
    """Make one cycle of an oscillation.

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

    if cycle_params[0] not in ['gaussian', 'exp', '2exp']:
        raise ValueError('Did not recognize cycle type.')

    if cycle_params[0] == 'gaussian':
        # cycle_params defines std in seconds
        cycle = signal.gaussian(t_ker * fs, cycle_params[1] * fs)

    elif cycle_params[0] == 'exp':
        # cycle_params defines decay time constant in seconds
        cycle = make_synaptic_kernel(t_ker, fs, 0, cycle_params[1])

    elif cycle_params[0] == '2exp':
        # cycle_params defines rise and decay time constant in seconds
        cycle = make_synaptic_kernel(t_ker, fs, cycle_params[1], cycle_params[2])

    return cycle


def make_synaptic_kernel(t_ker, fs, tau_r, tau_d):
    """Creates synaptic kernels that with specified time constants.

    Parameters
    ----------
    t_ker : float
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

    # NOTE: sometimes t_ker is not exact, resulting in a slightly longer or
    #   shorter times vector, which will affect final signal length
    #   https://docs.python.org/2/tutorial/floatingpoint.html
    times = np.arange(0, t_ker, 1./fs)

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
            warnings.warn('Rise time constant should be shorter than decay time constant.')

        # I(t)=(tau_r/(tau_r-tau_d))*(exp(-t/tau_d)-exp(-t/tau_r))
        kernel = (np.exp(-times / tau_d) - np.exp(-times / tau_r))

    # Normalize the integral to 1
    kernel = kernel / np.sum(kernel)

    return kernel
