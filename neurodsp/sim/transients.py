"""Simulating transients."""

from warnings import warn

import numpy as np
from scipy.stats import norm

from neurodsp.utils.data import create_times, compute_nsamples

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


def sim_skewed_gaussian(n_seconds, fs, center, std, alpha, height=1):
    """Simulate a skewed gaussian.

    Parameters
    ----------
    n_seconds : float
        Length of cycle window in seconds.
    fs : float
        Sampling frequency of the cycle simulation.
    center : float
        The center of the skewed gaussian.
    std : float
        Standard deviation of the gaussian kernel, in seconds.
    alpha : float
        Magnitiude and direction of the skew.
    center : float, optional, default: 0.5
        Time where the peak occurs in the pre-skewed gaussian.
    height : float, optional, default: 1.
        Maximum value of the cycle.

    Returns
    -------
    ys : 1d array
        Output values for skewed gaussian function.
    """

    # Prevent circular import
    from neurodsp.sim.cycles import sim_gaussian_cycle

    n_samples = compute_nsamples(n_seconds, fs)

    # Gaussian distribution
    cycle = sim_gaussian_cycle(n_seconds, fs, std, center)

    # Skewed cumulative distribution function.
    #   Assumes time are centered around 0. Adjust to center around 0.5.
    times = np.linspace(-1, 1, n_samples)
    cdf = norm.cdf(alpha * ((times - ((center * 2) -1 )) / std))

    # Skew the gaussian
    cycle = cycle * cdf

    # Rescale height
    cycle = (cycle / np.max(cycle)) * height

    return cycle


def sim_action_potential(n_seconds, fs, centers, stds, alphas, heights):
    """Simulate an action potential as the sum of two inverse, skewed gaussians.

    Parameters
    ----------
    n_seconds : float
        Length of cycle window in seconds.
    fs : float
        Sampling frequency of the cycle simulation.
    centers : array-like or float
        Times where the peak occurs in the pre-skewed gaussian.
    stds : array-like or float
        Standard deviations of the gaussian kernels, in seconds.
    alphas : array-like or float
        Magnitiude and direction of the skew.
    heights : array-like or float
        Maximum value of the cycles.

    Returns
    -------
    cycle : 1d array
        Simulated spike cycle.
    """

    # Determine number of parameters
    n_params = []

    if isinstance(centers, (tuple, list, np.ndarray)):
        n_params.append(len(centers))
    else:
        centers = repeat(centers)

    if isinstance(stds, (tuple, list, np.ndarray)):
        n_params.append(len(stds))
    else:
        stds = repeat(stds)

    if isinstance(heights, (tuple, list, np.ndarray)):
        n_params.append(len(heights))
    else:
        heights = repeat(heights)

    # Parameter checking
    if len(n_params) == 0:
        raise ValueError('Array-like expected for one of {centers, stds, heights}.')

    for param_len in n_params[1:]:
        if param_len != n_params[0]:
            raise ValueError('Unequal lengths between two or more of {centers, stds, heights}')

    # Initialize cycle array
    n_samples = compute_nsamples(n_seconds, fs)

    n_params = n_params[0]

    cycle = np.zeros((n_params, n_samples))

    # Simulate
    for idx, (center, std, alpha, height) in enumerate(zip(centers, stds, alphas, heights)):
        cycle[idx] = sim_skewed_gaussian(n_seconds, fs, center, std, alpha, height)

    cycle = np.sum(cycle, axis=0)

    return cycle
