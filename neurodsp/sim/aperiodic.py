"""Simulating time series, with aperiodic activity."""

import numpy as np
from scipy import signal
from scipy.stats import zscore

from neurodsp.filt import filter_signal, infer_passtype
from neurodsp.spectral import rotate_powerlaw
from neurodsp.sim.transients import make_synaptic_kernel

###################################################################################################
###################################################################################################

def sim_poisson_pop(n_seconds, fs, n_neurons, firing_rate):
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


def sim_synaptic_noise(n_seconds, fs, n_neurons=1000, firing_rate=2,
                       tau_r=0, tau_d=0.01, t_ker=None):
    """Simulate a neural signal with 1/f characteristics beyond a knee frequency.

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

    # Simulate an extra bit because the convolution will snip it
    sig = sim_poisson_pop((n_seconds + t_ker), fs, n_neurons, firing_rate)
    ker = make_synaptic_kernel(t_ker, fs, tau_r, tau_d)
    sig = np.convolve(sig, ker, 'valid')[:-1]

    return sig


def sim_ou_process(n_seconds, fs, theta=1., mu=0., sigma=5.):
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


def sim_variable_powerlaw(n_seconds, fs, exponent=-2.0):
    """Generate a power law time series with specified exponent by spectrally rotating white noise.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    exponent : float
        Desired power-law exponent: beta in P(f)=f^beta.

    Returns
    -------
    sig: 1d array
        Time-series with the desired power-law exponent.
    """

    n_samps = int(n_seconds * fs)
    sig = np.random.randn(n_samps)

    # Compute the FFT
    fft_output = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), 1. / fs)

    # Rotate spectrum and invert, zscore to normalize
    fft_output_rot = rotate_powerlaw(freqs, fft_output, exponent / 2.)
    sig = zscore(np.real(np.fft.ifft(fft_output_rot)))

    return sig


def sim_filtered_noise(n_seconds, fs, exponent=-2., f_range=(0.5, None), **filter_kwargs):
    """Simulate colored noise that is highpass or bandpass filtered.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    exponent : float, optional, default=-2
        Desired power-law exponent: beta in P(f)=f^beta. Negative exponent
        denotes decay (i.e., negative slope in log-log spectrum).
    f_range : 2-element array (lo, hi) or None, optional
        Frequency range of simulated data. If not provided, default to a highpass at 0.5 Hz.
    **filter_kwargs : kwargs, optional
        Keyword arguments to pass to `filter_signal`.

    Returns
    -------
    noise : 1d array
        Filtered noise.
    """

    noise = sim_variable_powerlaw(n_seconds, fs, exponent)
    noise = filter_signal(noise, fs, infer_passtype(f_range), f_range, **filter_kwargs)

    return noise


def get_aperiodic_sim(n_seconds, fs, generator, **noise_kwargs):
    """Get simulated aperiodic data, of the specified type.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    generator : {'filtered_powerlaw', 'powerlaw', 'synaptic', 'lorentzian', 'ou_process'}
        Generator for aperiodic activity, as one of the simulators in neurodsp.sim.

    Returns
    -------
    sig : 1d array
        Simulated aperiodic signal.
    """

    # Check that the specified aperiodic generator is valid
    if generator not in ['filtered_powerlaw', 'powerlaw', 'synaptic', 'lorentzian', 'ou_process']:
        raise ValueError('Did not recognize aperiodic generator type.\
                          Please check doc for acceptable function names.')

    if generator == 'filtered_powerlaw':
        sig = sim_filtered_noise(n_seconds, fs, **noise_kwargs)

    elif generator == 'powerlaw':
        sig = sim_variable_powerlaw(n_seconds, fs, **noise_kwargs)

    elif generator in ['synaptic', 'lorentzian']:
        sig = sim_synaptic_noise(n_seconds, fs, **noise_kwargs)

    elif generator == 'ou_process':
        sig = sim_ou_process(n_seconds, fs, **noise_kwargs)

    return sig
