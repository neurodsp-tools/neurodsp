"""Simulating time series, with aperiodic activity."""

import warnings

import numpy as np
from scipy.stats import zscore
from scipy import signal

from neurodsp.filt import filter_signal_fir
from neurodsp.spectral import rotate_powerlaw

###################################################################################################
###################################################################################################

def sim_poisson_pop(n_seconds, fs, n_neurons, firing_rate):
    """Simulates a poisson population.

    It is essentially white noise, but satisfies the Poisson property, i.e. mean(X) = var(X).

    The lambda parameter of the Poisson process (total rate) is determined as
    firing rate * number of neurons, i.e. summation of poisson processes is still
    a poisson processes.

    Note that the Gaussian approximation for a sum of Poisson processes is only
    a good approximation for large lambdas.

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
    """

    n_samples = int(n_seconds * fs)

    # poisson population rate signal scales with # of neurons and individual rate
    lam = n_neurons * firing_rate

    # variance is equal to the mean
    sig = np.random.normal(loc=lam, scale=lam**0.5, size=n_samples)

    # enforce that sig is non-negative in cases of low firing rate
    sig[np.where(sig < 0.)] = 0.

    return sig


def make_synaptic_kernel(t_ker, fs, tau_r, tau_d):
    """Creates synaptic kernels that with specified time constants.

    3 types of kernels are available, based on combinations of time constants:
    - tau_r == tau_d  : alpha (function) synapse
    - tau_r = 0       : instantaneous rise, (single) exponential decay
    - tau_r!=tau_d!=0 : double-exponential (rise and decay)

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
    """

    ### NOTE: sometimes t_ker is not exact, resulting in a slightly longer or
    ###     shorter times vector, which will affect final signal length
    # https://docs.python.org/2/tutorial/floatingpoint.html
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


def sim_synaptic_noise(n_seconds, fs, n_neurons=1000, firing_rate=2, tau_r=0, tau_d=0.01, t_ker=None):
    """Simulate a neural signal with 1/f characteristics beyond a knee frequency.

    The resulting signal is most similar to unsigned intracellular current or conductance change.

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

    Discretized Ornstein-Uhlenbeck process:
    dx = theta*(x-mu)*dt + sigma*dWt
    dWt : increments of Wiener process, i.e. white noise
    theta : memory scale (higher = faster fluc)
    mu : mean
    sigma : std

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

    # compute FFT
    fc = np.fft.fft(sig)
    f_axis = np.fft.fftfreq(len(sig), 1. / fs)

    # rotate spectrum and invert, zscore to normalize
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

    # # USE FILTER FROM NDSP
    # # Infer passtype & filter signal
    # #   TODO: Update this with infer_passtype helper function
    # if f_range[1] is None:
    #     pass_type = 'highpass'
    # else:
    #     pass_type = 'bandpass'
    # noise = filter_signal_fir(noise, fs, pass_type, f_range)

    # return noise


def _return_noise_sim(n_seconds, fs, noise_generator, noise_args):


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
