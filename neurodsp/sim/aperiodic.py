"""Simulating time series, with aperiodic activity."""

import numpy as np
from scipy.stats import zscore

from neurodsp.filt import filter_signal, infer_passtype
from neurodsp.filt.fir import compute_filter_length
from neurodsp.filt.checks import check_filter_definition
from neurodsp.utils import remove_nans
from neurodsp.utils.data import create_times
from neurodsp.utils.decorators import normalize
from neurodsp.spectral import rotate_powerlaw
from neurodsp.sim.transients import sim_synaptic_kernel

###################################################################################################
###################################################################################################

@normalize
def sim_poisson_pop(n_seconds, fs, n_neurons=1000, firing_rate=2):
    """Simulate a Poisson population.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    n_neurons : int, optional, default: 1000
        Number of neurons in the simulated population.
    firing_rate : float, optional, default: 2
        Firing rate of individual neurons in the population.

    Returns
    -------
    sig : 1d array
        Simulated population activity.

    Notes
    -----
    The simulated signal is essentially white noise, but satisfies the Poisson
    property, i.e. mean(X) = var(X).

    The lambda parameter of the Poisson process (total rate) is determined as
    firing rate * number of neurons, i.e. summation of Poisson processes is still
    a Poisson processes.

    Note that the Gaussian approximation for a sum of Poisson processes is only
    a good approximation for large lambdas.

    Examples
    --------
    Simulate a Poisson population:

    >>> sig = sim_poisson_pop(n_seconds=1, fs=500, n_neurons=1000, firing_rate=2)
    """

    # Poisson population rate signal scales with # of neurons and individual rate
    lam = n_neurons * firing_rate

    # Variance is equal to the mean
    sig = np.random.normal(loc=lam, scale=lam**0.5, size=int(n_seconds * fs))

    # Enforce that sig is non-negative in cases of low firing rate
    sig[np.where(sig < 0.)] = 0.

    return sig


@normalize
def sim_synaptic_current(n_seconds, fs, n_neurons=1000, firing_rate=2.,
                         tau_r=0., tau_d=0.01, t_ker=None):
    """Simulate a signal as a synaptic current, which has 1/f characteristics with a knee.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    n_neurons : int, optional, default: 1000
        Number of neurons in the simulated population.
    firing_rate : float, optional, default: 2
        Firing rate of individual neurons in the population.
    tau_r : float, optional, default: 0.
        Rise time of synaptic kernel, in seconds.
    tau_d : float, optional, default: 0.01
        Decay time of synaptic kernel, in seconds.
    t_ker : float, optional
        Length of time of the simulated synaptic kernel, in seconds.

    Returns
    -------
    sig : 1d array
        Simulated synaptic current.

    Notes
    -----
    The resulting signal is most similar to unsigned intracellular current or conductance change.

    Examples
    --------
    Simulate a synaptic current signal:

    >>> sig = sim_synaptic_current(n_seconds=1, fs=500)
    """

    # If not provided, compute t_ker as a function of decay time constant
    if t_ker is None:
        t_ker = 5. * tau_d

    # Simulate an extra bit because the convolution will trim & turn off normalization
    sig = sim_poisson_pop((n_seconds + t_ker), fs, n_neurons, firing_rate,
                          mean=None, variance=None)
    ker = sim_synaptic_kernel(t_ker, fs, tau_r, tau_d)
    sig = np.convolve(sig, ker, 'valid')[:-1]

    return sig


@normalize
def sim_random_walk(n_seconds, fs, theta=1., mu=0., sigma=5.):
    """Simulate a mean-reverting random walk, as an Ornstein-Uhlenbeck process.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    theta : float, optional, default: 1.0
        Memory scale parameter. Larger theta values create faster fluctuations.
    mu : float, optional, default: 0.0
        Mean of the random walk.
    sigma : float, optional, default: 5.0
        Standard deviation of the random walk.

    Returns
    -------
    sig: 1d array
        Simulated random walk signal.

    Notes
    -----
    The random walk is simulated as a discretized Ornstein-Uhlenbeck process:

    `dx = theta*(x-mu)*dt + sigma*dWt`

    Where:

    - mu : mean
    - sigma : standard deviation
    - theta : memory scale
    - dWt : increments of Wiener process, i.e. white noise

    References
    ----------
    See the wikipedia page for the integral solution:

    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process#Formal_solution

    Examples
    --------
    Simulate a Ornstein-Uhlenbeck random walk:

    >>> sig = sim_random_walk(n_seconds=1, fs=500, theta=1.)
    """

    times = create_times(n_seconds, fs)

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
    """Simulate a power law time series, with a specified exponent.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    exponent : float, optional, default: -2
        Desired power-law exponent, of the form P(f)=f^exponent.
    f_range : list of [float, float] or None, optional
        Frequency range to filter simulated data, as [f_lo, f_hi], in Hz.
    **filter_kwargs : kwargs, optional
        Keyword arguments to pass to `filter_signal`.

    Returns
    -------
    sig: 1d array
        Time-series with the desired power law exponent.

    Examples
    --------
    Simulate a power law signal, with an exponent of -2 (brown noise):

    >>> sig = sim_powerlaw(n_seconds=1, fs=500, exponent=-2.0)

    Simulate a power law signal, with a highpass filter applied at 2 Hz:

    >>> sig = sim_powerlaw(n_seconds=1, fs=500, exponent=-1.5, f_range=(2, None))
    """

    # Get the number of samples to simulate for the signal
    #   If signal is to be filtered, with FIR, add extra to compensate for edges
    if f_range and filter_kwargs.get('filter_type', None) != 'iir':

        pass_type = infer_passtype(f_range)
        filt_len = compute_filter_length(fs, pass_type,
                                         *check_filter_definition(pass_type, f_range),
                                         n_seconds=filter_kwargs.get('n_seconds', None),
                                         n_cycles=filter_kwargs.get('n_cycles', 3))

        n_samples = int(n_seconds * fs) + filt_len + 1

    else:
        n_samples = int(n_seconds * fs)

    sig = _create_powerlaw(n_samples, fs, exponent)

    if f_range is not None:
        sig = filter_signal(sig, fs, infer_passtype(f_range), f_range,
                            remove_edges=True, **filter_kwargs)
        # Drop the edges, that were compensated for, if not using FIR filter
        if not filter_kwargs.get('filter_type', None) == 'iir':
            sig, _ = remove_nans(sig)

    return sig


def _create_powerlaw(n_samples, fs, exponent):
    """Create a power law time series.

    Parameters
    ----------
    n_samples : int
        The number of samples to simulate.
    fs : float
        Sampling rate of simulated signal, in Hz.
    exponent : float
        Desired power-law exponent, of the form P(f)=f^exponent.

    Returns
    -------
    sig: 1d array
        Time-series with the desired power law exponent.

    Notes
    -----
    This function creates variable power law exponents by spectrally rotating white noise.
    """

    # Start with white noise signal, that we will rotate, in frequency space
    sig = np.random.randn(n_samples)

    # Compute the FFT
    fft_output = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), 1. / fs)

    # Rotate spectrum and invert back to time series, with a z-score to normalize
    #   Delta exponent is divided by two, as the FFT output is in units of amplitude not power
    fft_output_rot = rotate_powerlaw(freqs, fft_output, -exponent/2)
    sig = zscore(np.real(np.fft.ifft(fft_output_rot)))

    return sig
