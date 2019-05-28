"""Simulating time series, with aperiodic activity."""

import numpy as np
from scipy.stats import zscore

from neurodsp.filt import filter_signal, infer_passtype
from neurodsp.spectral import rotate_powerlaw
from neurodsp.utils.decorators import normalize
from neurodsp.sim.transients import sim_synaptic_kernel

###################################################################################################
###################################################################################################

@normalize
def sim_poisson_pop(n_seconds, fs, n_neurons=1000, firing_rate=2):
    """Simulates a poisson population.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    n_neurons : int
        Number of neurons in the simulated population.
    firing_rate : type
        Firing rate of individual neurons in the population.

    Returns
    -------
    sig : 1d array
        Simulated population activity.

    Notes
    -----
    It is essentially white noise, but satisfies the Poisson property, i.e. mean(X) = var(X).

    The lambda parameter of the Poisson process (total rate) is determined as
    firing rate * number of neurons, i.e. summation of poisson processes is still
    a poisson processes.

    Note that the Gaussian approximation for a sum of Poisson processes is only
    a good approximation for large lambdas.
    """

    # Poisson population rate signal scales with # of neurons and individual rate
    lam = n_neurons * firing_rate

    # Variance is equal to the mean
    sig = np.random.normal(loc=lam, scale=lam**0.5, size=int(n_seconds * fs))

    # Enforce that sig is non-negative in cases of low firing rate
    sig[np.where(sig < 0.)] = 0.

    return sig


@normalize
def sim_synaptic_current(n_seconds, fs, n_neurons=1000, firing_rate=2,
                         tau_r=0, tau_d=0.01, t_ker=None):
    """Simulate a neural signal as synaptic current, which has 1/f characteristics with a knee.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    n_neurons : int
        Number of neurons in the simulated population.
    firing_rate : float
        Firing rate of individual neurons in the population.
    tau_r : float
        Rise time of synaptic kernel, in seconds.
    tau_d : fload
        Decay time of synaptic kernel, in seconds.

    Returns
    -------
    sig : 1d array
        Simulated signal.

    Notes
    -----
    The resulting signal is most similar to unsigned intracellular current or conductance change.
    """

    # If not provided, compute t_ker as a function of decay time constant
    if t_ker is None:
        t_ker = 5. * tau_d

    # Simulate an extra bit because the convolution will snip it. Turn off normalization for this sig
    sig = sim_poisson_pop((n_seconds + t_ker), fs, n_neurons, firing_rate, mean=None, variance=None)
    ker = sim_synaptic_kernel(t_ker, fs, tau_r, tau_d)
    sig = np.convolve(sig, ker, 'valid')[:-1]

    return sig


@normalize
def sim_random_walk(n_seconds, fs, theta=1., mu=0., sigma=5.):
    """Simulate mean-reverting random walk, as an Ornstein-Uhlenbeck process.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    theta : float
        Memory scale - larger theta = faster fluctuation.
    mu : float
        Mean.
    sigma : float
        Standard deviation.

    Returns
    -------
    sig: 1d array
        Simulated signal.

    Notes
    -----
    Discretized Ornstein-Uhlenbeck process:
    dx = theta*(x-mu)*dt + sigma*dWt
    dWt : increments of Wiener process, i.e. white noise
    theta : memory scale (higher = faster fluc)
    mu : mean
    sigma : std

    References
    ----------
    See for integral solution:
        https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process#Solution
    """

    times = np.arange(0, n_seconds, 1 / fs)
    x0 = mu
    dt = times[1] - times[0]
    ws = np.random.normal(size=len(times))
    ex = np.exp(-theta * times)
    ws[0] = 0.
    sig = x0 * ex + mu * (1. - ex) + sigma * ex * \
        np.cumsum(np.exp(theta * times) * np.sqrt(dt) * ws)

    return sig


@normalize
def sim_powerlaw(n_seconds, fs, exponent=-2.0, f_range=None, **filter_kwargs):
    """Generate a power law time series with specified exponent by spectrally rotating white noise.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    exponent : float
        Desired power-law exponent, of the form P(f)=f^exponent.
    f_range : list of [float, float] or None, optional
        Frequency range to filter simulated data, as [f_lo, f_hi], in Hz.
    **filter_kwargs : kwargs, optional
        Keyword arguments to pass to `filter_signal`.

    Returns
    -------
    sig: 1d array
        Time-series with the desired power-law exponent.
    """

    n_samples = int(n_seconds * fs)
    sig = np.random.randn(n_samples)

    # Compute the FFT
    fft_output = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), 1. / fs)

    # Rotate spectrum and invert, zscore to normalize.
    #   Note: the delta exponent to be applied is divided by two, as
    #     the FFT output is in units of amplitude not power
    fft_output_rot = rotate_powerlaw(freqs, fft_output, -exponent/2)
    sig = zscore(np.real(np.fft.ifft(fft_output_rot)))

    if f_range is not None:
        filter_signal(sig, fs, infer_passtype(f_range), f_range, **filter_kwargs)

    return sig
