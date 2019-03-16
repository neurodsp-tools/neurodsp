"""Simulating time series, with aperiodic activity."""

import numpy as np
from scipy import signal
from scipy.stats import zscore

from neurodsp.filt import filter_signal_fir
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
    sig = sim_poisson_pop(n_seconds=(n_seconds + t_ker),
                          fs=fs, n_neurons=n_neurons, firing_rate=firing_rate)
    ker = make_synaptic_kernel(t_ker=t_ker, fs=fs, tau_r=tau_r, tau_d=tau_d)
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
    fc = np.fft.fft(sig)
    f_axis = np.fft.fftfreq(len(sig), 1. / fs)

    # Rotate spectrum and invert, zscore to normalize
    fc_rot = rotate_powerlaw(
        f_axis, fc, exponent / 2., f_rotation=None)
    sig = zscore(np.real(np.fft.ifft(fc_rot)))

    return sig


def sim_filtered_noise(n_seconds, fs, exponent=-2., f_range=(0.5, None), filter_order=None):
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
    filter_order : int, optional
        Order of filter. If not provided, defaults to 3 times the highpass filter cycle length.

    Returns
    -------
    noise : 1d array
        Filtered noise.
    """

    # Simulate colored noise
    noise = sim_variable_powerlaw(n_seconds, fs, exponent)

    nyq = fs / 2.

    # Determine order of highpass filter (3 cycles of f_hipass)
    if filter_order is None:
        filter_order = int(3 * fs / f_range[0])

    # High pass filtered
    if f_range[1] is None:
        # Make filter order odd if necessary
        if filter_order % 2 == 0:
            #print('NOTE: Increased high-pass filter order by 1 in order to be odd')
            filter_order += 1

        # High pass filter
        taps = signal.firwin(filter_order, f_range[0] / nyq, pass_zero=False)
        noise = signal.filtfilt(taps, [1], noise)

    # Band pass filtered
    else:
        taps = signal.firwin(filter_order, np.array(
            f_range) / nyq, pass_zero=False)
        noise = signal.filtfilt(taps, [1], noise)

    return noise


def _return_noise_sim(n_seconds, fs, noise_generator, noise_args):
    """   """

    if isinstance(noise_generator, str):

        # Check that the specified noise generator is valid
        valid_noise_generators = ['filtered_powerlaw', 'powerlaw', 'synaptic', 'lorentzian', 'ou_process']
        if noise_generator not in valid_noise_generators:
            raise ValueError('Did not recognize noise type. Please check doc for acceptable function names.')


        if noise_generator == 'filtered_powerlaw':
            noise = sim_filtered_noise(n_seconds, fs, **noise_args)

        elif noise_generator == 'powerlaw':
            noise = sim_variable_powerlaw(n_seconds, fs, **noise_args)

        elif noise_generator == 'synaptic' or noise_generator == 'lorentzian':
            noise = sim_synaptic_noise(n_seconds, fs, **noise_args)

        elif noise_generator == 'ou_process':
            noise = sim_ou_process(n_seconds, fs, **noise_args)


    elif isinstance(noise_generator, np.ndarray):
        if len(noise_generator) != int(n_seconds * fs):
            raise ValueError('Custom noise is not of same length as required oscillation length.')
        else:
            noise = noise_generator

    else:
        raise ValueError('Unsupported noise type: must be np.ndarray or str.')

    return noise
