"""Simulating transients."""

from warnings import warn

import numpy as np

from neurodsp.utils.data import create_times

###################################################################################################
###################################################################################################

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
