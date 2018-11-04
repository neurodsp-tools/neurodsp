"""Simulating time series, including oscillations and aperiodic backgrounds."""

import warnings

import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal

from neurodsp import spectral

###################################################################################################
###################################################################################################


def sim_oscillator(n_seconds, fs, freq, rdsym=.5):
    """Simulate an oscillation.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds
    fs : float
        Signal sampling rate, in Hz
    freq : float
        Oscillator frequency
    rdsym : float
        Rise-decay symmetry of the oscillator, as fraction of the period in the rise time
            =0.5 - symmetric (sine wave)
            <0.5 - shorter rise, longer decay
            >0.5 - longer rise, shorter decay

    Returns
    -------
    osc : 1d array
        Oscillating time series
    """

    # Compute number of samples per cycle and number of cycles
    n_samples_cycle = int(np.ceil(fs / freq))
    n_samples = int(fs * n_seconds)
    n_cycles = int(np.ceil(n_seconds * freq))

    # Determine number of samples in rise and decay periods
    rise_samples = int(np.round(n_samples_cycle * rdsym))
    decay_samples = n_samples_cycle - rise_samples

    # Make phase array for a single cycle, then repeat it
    pha_one_cycle = np.hstack([np.linspace(
        0, np.pi, decay_samples + 1), np.linspace(-np.pi, 0, rise_samples + 1)[1:-1]])
    phase_t = np.tile(pha_one_cycle, n_cycles)
    phase_t = phase_t[:n_samples]

    # Transform phase into an oscillator
    osc = np.cos(phase_t)

    return osc


def sim_noisy_oscillator(n_seconds, fs, freq, noise_generator, noise_args, rdsym=.5, ratio_osc_var=1):
    """Simulate an oscillation embedded in background 1/f noise.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds
    fs : float
        Signal sampling rate, in Hz
    freq : float
        Oscillator frequency
    noise_generator: str or numpy.ndarray
        Noise model, can be one of the simulators in neurodsp.sim specificed as a string, or a custom
        numpy.ndarray with the same number of samples as the oscillation (n_seconds*fs).
        Possible models (see respective documentation):
            - 'filtered_powerlaw': sim.sim_filtered_noise()
            - 'powerlaw': sim.sim_variable_powerlaw()
            - 'synaptic' or 'lorentzian': sim.sim_synaptic_noise()
            - 'ou_process': sim.sim_ou_process()
    noise_args: dict('argname':argval, ...)
        Function arguments for the neurodsp.sim noise generaters. See API for arg names.
        NOTE: all args, including optional ones, are required, EXCEPT n_seconds and fs.
    rdsym : float
        Rise-decay symmetry of the oscillator, as fraction of the period in the rise time;
            =0.5 - symmetric (i.e., sine wave, default)
            <0.5 - shorter rise, longer decay
            >0.5 - longer rise, shorter decay
    ratio_osc_var : float
        Ratio of oscillator variance to noise variance
            >1 - oscillator is stronger
            <1 - noise is stronger
            Defaults to 1.

    Returns
    -------
    osc: 1d array
        Oscillator with noise
    """

    # Determine length of signal in samples
    n_samples = int(n_seconds * fs)

    # Generate noise
    noise = _return_noise_sim(n_seconds, fs, noise_generator, noise_args)

    # Generate oscillator
    oscillator = sim_oscillator(n_seconds, fs, freq, rdsym=rdsym)

    # Normalize by variance
    oscillator_var = np.var(oscillator)
    noise_var = np.var(noise)
    noise = np.sqrt(noise**2 * oscillator_var /
                    (noise_var * ratio_osc_var)) * np.sign(noise)

    # Combine oscillator and noise
    osc = oscillator + noise

    return osc


def sim_bursty_oscillator(n_seconds, fs, freq, rdsym=.5, prob_enter_burst=.2,
                          prob_leave_burst=.2, cycle_features=None,
                          return_cycle_df=False):
    """Simulate a bursty oscillation.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds
    fs : float
        Sampling rate of simulated signal, in Hz
    freq : float
        Oscillator frequency, in Hz
    rdsym : float
        Rise-decay symmetry of the oscillator, as fraction of the period in the rise time;
            =0.5 - symmetric (sine wave)
            <0.5 - shorter rise, longer decay
            >0.5 - longer rise, shorter decay
    prob_enter_burst : float
        Rrobability of a cycle being oscillating given the last cycle is not oscillating
    prob_leave_burst : float
        Probability of a cycle not being oscillating given the last cycle is oscillating
    cycle_features : dict
        specify the mean and standard deviations
        (within and across bursts) of each cycle's
        amplitude, period, and rise-decay symmetry.
        This can include a complete or incomplete set
        (using defaults) of the following keys:
        amp_mean - mean cycle amplitude
        amp_std - standard deviation of cycle amplitude
        amp_burst_std - std. of mean amplitude for each burst
        period_mean - mean period (computed from `freq`)
        period_std - standard deviation of period (samples)
        period_burst_std - std. of mean period for each burst
        rdsym_mean - mean rise-decay symmetry
        rdsym_std - standard deviation of rdsym
        rdsym_burst_std - std. of mean rdsym for each burst
    return_cycle_df : bool
        if True, return the dataframe that contains the simulation
        parameters for each cycle. This may be useful for computing
        power, for example. Because the power of the oscillator
        should only be considered over the times where there's
        bursts, not when there's nothing.

    Returns
    -------
    sig : 1d array
        bursty oscillator
    df : pd.DataFrame
        cycle-by-cycle properties of the simulated oscillator
    """

    # Define default parameters for cycle features
    mean_period_samples = int(fs / freq)
    cycle_features_use = {'amp_mean': 1, 'amp_burst_std': 0, 'amp_std': 0,
                          'period_mean': mean_period_samples, 'period_burst_std': 0, 'period_std': 0,
                          'rdsym_mean': rdsym, 'rdsym_burst_std': 0, 'rdsym_std': 0}

    # Overwrite default cycle features with those specified
    if cycle_features is not None:
        for k in cycle_features:
            cycle_features_use[k] = cycle_features[k]

    # Determine number of cycles to generate
    n_samples = n_seconds * fs
    n_cycles_overestimate = int(np.ceil(n_samples / mean_period_samples * 2))

    # Simulate if a series of cycles are oscillating or not oscillating
    is_oscillating = [False]
    n_cycles_current = 1
    while n_cycles_current < n_cycles_overestimate:
        rand_num = np.random.rand()
        if is_oscillating[-1]:
            is_oscillating.append(rand_num > prob_leave_burst)
        else:
            is_oscillating.append(rand_num < prob_enter_burst)
        n_cycles_current += 1

    # Determine period, amp, and rdsym for each cycle
    periods = []
    amps = []
    rdsyms = []
    for is_osc in is_oscillating:
        if is_osc is False:
            period = cycle_features_use['period_mean'] + \
                np.random.randn() * cycle_features_use['period_std']
            periods.append(int(period))
            amps.append(np.nan)
            rdsyms.append(np.nan)

            current_burst_period_mean = np.nan
            current_burst_amp_mean = np.nan
            current_burst_rdsym_mean = np.nan
        else:
            if np.isnan(current_burst_period_mean):
                current_burst_period_mean = cycle_features_use['period_mean'] + \
                    np.random.randn() * cycle_features_use['period_burst_std']
                current_burst_amp_mean = cycle_features_use['amp_mean'] + \
                    np.random.randn() * cycle_features_use['amp_burst_std']
                current_burst_rdsym_mean = cycle_features_use['rdsym_mean'] + \
                    np.random.randn() * cycle_features_use['rdsym_burst_std']
            period = current_burst_period_mean + \
                np.random.randn() * cycle_features_use['period_std']
            amp = current_burst_amp_mean + \
                np.random.randn() * cycle_features_use['amp_std']
            rdsym = current_burst_rdsym_mean + \
                np.random.randn() * cycle_features_use['rdsym_std']
            periods.append(int(period))
            amps.append(amp)
            rdsyms.append(rdsym)

    df = pd.DataFrame({'is_cycle': is_oscillating, 'period': periods,
                       'amp': amps, 'rdsym': rdsyms})
    df['start_sample'] = np.insert(df['period'].cumsum().values[:-1], 0, 0)
    df = df[df['start_sample'] < n_samples]

    # Shorten df to only cycles that are included in the data

    # Simulate time series for each cycle
    sig = np.array([])
    last_cycle_oscillating = False
    for ind, row in df.iterrows():
        if row['is_cycle'] is False:
            # If last cycle was oscillating, add a decay to 0 then 0s
            if last_cycle_oscillating:
                decay_pha = np.linspace(0, np.pi / 2, int(row['period'] / 4))
                decay_t = np.cos(decay_pha) * sig[-1]
                sig = np.append(sig, decay_t)

                cycle_t = np.zeros(row['period'] - int(row['period'] / 4))
                sig = np.append(sig, cycle_t)
            else:
                # Add a blank cycle
                cycle_t = np.zeros(row['period'])
                sig = np.append(sig, cycle_t)
            last_cycle_oscillating = False
        else:
            # If last cycle was oscillating, add a decay to 0
            if not last_cycle_oscillating:
                rise_pha = np.linspace(-np.pi / 2, 0,
                                       int(row['period'] / 4))[1:]
                rise_t = np.cos(rise_pha) * row['amp']
                sig[-len(rise_t):] = rise_t

            # Add a cycle with rdsym
            rise_samples = int(np.round(row['period'] * row['rdsym']))
            decay_samples = row['period'] - rise_samples
            pha_t = np.hstack([np.linspace(0, np.pi, decay_samples + 1)[1:],
                               np.linspace(-np.pi, 0, rise_samples + 1)[1:]])
            cycle_t = np.cos(pha_t)

            # Adjust decay if the last cycle was oscillating
            if last_cycle_oscillating:
                scaling = (row['amp'] + sig[-1]) / 2
                offset = (sig[-1] - row['amp']) / 2
                cycle_t[:decay_samples] = cycle_t[:decay_samples] * \
                    scaling + offset
                cycle_t[decay_samples:] = cycle_t[decay_samples:] * row['amp']
            else:
                cycle_t = cycle_t * row['amp']
            sig = np.append(sig, cycle_t)
            last_cycle_oscillating = True
    sig = sig[:n_samples]

    if return_cycle_df:
        # Remove last row of df
        df.drop(df.index[len(df) - 1], inplace=True)
        return sig, df
    else:
        return sig


def sim_noisy_bursty_oscillator(n_seconds, fs, freq, noise_generator, noise_args, rdsym=.5,
    ratio_osc_var=1, prob_enter_burst=.2, prob_leave_burst=.2, cycle_features=None,
        return_components=False, return_cycle_df=False):

    """Simulate a bursty oscillation embedded in background 1/f noise.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds
    fs : float
        Sampling rate of simulated signal, in Hz
    freq : float
        Oscillator frequency, in Hz
    noise_generator: str or numpy.ndarray
        Noise model, can be one of the simulators in neurodsp.sim specificed as a string, or a custom
        numpy.ndarray with the same number of samples as the oscillation (n_seconds*fs).
        Possible models (see respective documentation):
            - 'filtered_powerlaw': sim.sim_filtered_noise()
            - 'powerlaw': sim.sim_variable_powerlaw()
            - 'synaptic' or 'lorentzian': sim.sim_synaptic_noise()
            - 'ou_process': sim.sim_ou_process()
    noise_args: dict('argname':argval, ...)
        Function arguments for the neurodsp.sim noise generaters. See API for arg names.
        NOTE: all args, including optional ones, are required, EXCEPT n_seconds and fs.
    rdsym : float
        Rise-decay symmetry of the oscillator as fraction of the period in the rise time
            =0.5 - symmetric (sine wave)
            <0.5 - shorter rise, longer decay
            >0.5 - longer rise, shorter decay
    ratio_osc_var : float
        Ratio of oscillator power to noise power
            >1 - oscillator is stronger
            <1 - noise is stronger
    prob_enter_burst : float
        Probability of a cycle being oscillating given the last cycle is not oscillating
    prob_leave_burst : float
        Probability of a cycle not being oscillating given the last cycle is oscillating
    cycle_features : dict
        specify the mean and standard deviations
        (within and across bursts) of each cycle's
        amplitude, period, and rise-decay symmetry.
        This can include a complete or incomplete set
        (using defaults) of the following keys:
        amp_mean - mean cycle amplitude
        amp_std - standard deviation of cycle amplitude
        amp_burst_std - std. of mean amplitude for each burst
        period_mean - mean period (computed from `freq`)
        period_std - standard deviation of period (samples)
        period_burst_std - std. of mean period for each burst
        rdsym_mean - mean rise-decay symmetry
        rdsym_std - standard deviation of rdsym
        rdsym_burst_std - std. of mean rdsym for each burst
    return_components: bool
        Whether to return the oscillator and noise separately, in addition to the signal
    return_cycle_df : bool
        if True, return the dataframe that contains the simulation
        parameters for each cycle. This may be useful for computing
        power, for example. Because the power of the oscillator
        should only be considered over the times where there's
        bursts, not when there's nothing.

    Returns
    -------
    signal : np.array
        bursty oscillator with noise time series
    oscillator : np.array
        bursty oscillator component of signal
    noise : np.array
        noise component of signal
    df : pd.DataFrame
        cycle-by-cycle properties of the simulated oscillator
    """

    # Generate noise
    noise = _return_noise_sim(n_seconds, fs, noise_generator, noise_args)

    # Generate oscillator
    oscillator, df = sim_bursty_oscillator(n_seconds, fs, freq, rdsym=rdsym,
                                           prob_enter_burst=prob_enter_burst,
                                           prob_leave_burst=prob_leave_burst,
                                           cycle_features=cycle_features,
                                           return_cycle_df=True)

    # Determine samples of burst so that can compute signal power over only those times
    is_osc = np.zeros(len(oscillator), dtype=bool)
    for ind, row in df.iterrows():
        if row['is_cycle']:
            is_osc[row['start_sample']:row['start_sample'] + row['period']] = True

    # Normalize noise power
    oscillator_var = np.var(oscillator[is_osc])
    noise_var = np.var(noise)
    noise = np.sqrt(noise**2 * oscillator_var /
                    (noise_var * ratio_osc_var)) * np.sign(noise)

    # Combine oscillator and noise
    signal = oscillator + noise

    if return_components:
        if return_cycle_df:
            return signal, oscillator, noise, df
        return signal, oscillator, noise
    else:
        if return_cycle_df:
            return signal, df
        return signal


def _return_noise_sim(n_seconds, fs, noise_generator, noise_args):
    # Generate noise
    if type(noise_generator) is str:
        # use neurodsp defined noise generators
        if noise_generator == 'filtered_powerlaw':
            noise = sim_filtered_noise(
                n_seconds, fs, noise_args['exponent'], noise_args['f_range'], noise_args['filter_order'])

        elif noise_generator == 'powerlaw':
            noise = sim_variable_powerlaw(n_seconds, fs, noise_args['exponent'])

        elif noise_generator == 'synaptic' or noise_generator == 'lorentzian':
            noise = sim_synaptic_noise(
                n_seconds, fs, noise_args['n_neurons'], noise_args['firing_rate'], noise_args['t_ker'], noise_args['tau_r'], noise_args['tau_d'])

        elif noise_generator == 'ou_process':
            noise = sim_ou_process(
                n_seconds, fs, noise_args['theta'], noise_args['mu'], noise_args['sigma'])

        else:
            raise ValueError(
                'Did not recognize noise type. Please check doc for acceptable function names.')

    elif type(noise_generator) is np.ndarray:
        if len(noise_generator) != int(n_seconds * fs):
            raise ValueError('Custom noise is not of same length as required oscillation length.')
        else:
            noise = noise_generator

    else:
        raise ValueError('Unsupported noise type: must be np.ndarray or str.')

    return noise


def sim_jittered_oscillator(n_seconds, fs, freq, jitter=0, cycle=('gaussian', 0.01)):
    """Simulate a jittered oscillator, as defined by the oscillator frequency,
    the oscillator cycle, and how much (in time) to jitter each period.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds
    fs : float
        Sampling rate of simulated signal, in Hz
    freq : float
        Frequency of simulated oscillator, in Hz
    jitter : float
        Maximum jitter of oscillation period, in seconds
    cycle : tuple or 1d array
        Oscillation cycle used in the simulation.
        If array, it's used directly.
        If tuple, it is generated based on given parameters.
        Possible values:
        ('gaussian', std): gaussian cycle, standard deviation in seconds
        ('exp', decay time): exponential decay, decay time constant in seconds
        ('2exp', rise time, decay time): exponential rise and decay

    Returns
    -------
    sig: 1d array
        Simulated oscillation with jitter.
    """

    # If cycle is a tuple, generate the window with given params,
    if isinstance(cycle, tuple):

        # defaults to 1 second window for a cycle, which is more than enough
        # if interested in longer period oscillations, just pass in premade cycle
        osc_cycle = make_osc_cycle(1, fs, cycle)

    # If cycle is an array, just use it to do the convolution
    else:
        osc_cycle = cycle

    # Binary "spike-train" of when each cycle should occur
    spks = np.zeros(int(n_seconds * fs + len(osc_cycle)) - 1)
    osc_period = int(fs / freq)

    # Generate oscillation "event" indices
    spk_indices = np.arange(osc_period, len(spks), osc_period)

    # Add jitter to "spike" indices
    if jitter != 0:

        spk_indices = spk_indices + \
            np.random.randint(low=-int(fs * jitter),
                              high=int(fs * jitter), size=len(spk_indices))

    spks[spk_indices] = 1
    sig = np.convolve(spks, osc_cycle, 'valid')

    return sig


def make_osc_cycle(t_ker, fs, cycle_params):
    """Make 1 cycle of oscillation.

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
        Defines the parameters for the oscillation cycle.
        Possible values:
            ('gaussian', std): gaussian cycle, standard deviation in seconds
            ('exp', decay time): exponential decay, decay time constant in seconds
            ('2exp', rise time, decay time): exponential rise and decay

    Returns
    -------
    cycle: 1d array
        Simulated oscillation cycle
    """

    if cycle_params[0] == 'gaussian':
        # cycle_params defines std in seconds
        cycle = signal.gaussian(t_ker * fs, cycle_params[1] * fs)

    elif cycle_params[0] == 'exp':
        # cycle_params defines decay time constant in seconds
        cycle = make_synaptic_kernel(t_ker, fs, 0, cycle_params[1])

    elif cycle_params[0] == '2exp':
        # cycle_params defines rise and decay time constant in seconds
        cycle = make_synaptic_kernel(
            t_ker, fs, cycle_params[1], cycle_params[2])

    else:
        raise ValueError('Did not recognize cycle type.')
    return cycle

# Noise (or, aperiodic background) simulators


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
        Simulation time, in seconds
    fs : float
        Sampling rate of simulated signal, in Hz
    n_neurons : int
        Number of neurons in the simulated population
    firing_rate : type
        Firing rate of individual neurons in the population

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
    kernel : array_like
        Computed synaptic kernel with length equal to t
    """

    times = np.arange(0, t_ker, 1 / fs)

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
            warnings.warn(
                'Rise time constant should be shorter than decay time constant.')

        # I(t)=(tau_r/(tau_r-tau_d))*(exp(-t/tau_d)-exp(-t/tau_r))
        kernel = (np.exp(-times / tau_d) - np.exp(-times / tau_r))

    # Normalize the integral to 1
    kernel = kernel / np.sum(kernel)

    return kernel


def sim_synaptic_noise(n_seconds, fs, n_neurons=1000, firing_rate=2, t_ker=1., tau_r=0, tau_d=0.01):
    """Simulate a neural signal with 1/f characteristics beyond a knee frequency.

    The resulting signal is most similar to unsigned intracellular current or conductance change.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds
    fs : float
        Sampling rate of simulated signal, in Hz
    n_neurons : int
        Number of neurons in the simulated population
    firing_rate : float
        Firing rate of individual neurons in the population
    t_ker : float
        Length of simulated kernel in seconds. Usually 1 second will suffice.
    tau_r : float
        Rise time of synaptic kernel, in seconds.
    tau_d : fload
        Decay time of synaptic kernel, in seconds.

    Returns
    -------
    sig : array_like (1D)
        Simulated signal.
    """

    # Simulate an extra bit because the convolution will snip it
    sig = sim_poisson_pop(n_seconds=(n_seconds + t_ker),
                          fs=fs, n_neurons=n_neurons, firing_rate=firing_rate)
    ker = make_synaptic_kernel(t_ker=t_ker, fs=fs, tau_r=tau_r, tau_d=tau_d)
    sig = np.convolve(sig, ker, 'valid')[:-1]

    return sig


def sim_ou_process(n_seconds, fs, theta=1., mu=0., sigma=5.):
    """Simulate mean-reverting random walk (Ornstein-Uhlenbeck process)

    Discretized Ornstein-Uhlenbeck process:
    dx = theta*(x-mu)*dt + sigma*dWt
    dWt : increments of Wiener process, i.e. white noise
    theta : memory scale (higher = faster fluc)
    mu : mean
    sigma : std


    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds
    fs : float
        Sampling rate of simulated signal, in Hz
    theta : float
        Memory scale - larger theta = faster fluctuation
    mu : float
        Mean
    sigma : float
        Standard deviation

    Returns
    -------
    sig: 1d array
        Simulated signal

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


def sim_variable_powerlaw(n_seconds, fs, exponent):
    """Generate a power law time series with specified exponent by spectrally rotating white noise.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds
    fs : float
        Sampling rate of simulated signal, in Hz
    exponent : float
        Desired power-law exponent - beta in P(f)=f^beta

    Returns
    -------
    sig: 1d array
        Time-series with the desired power-law exponent
    """

    n_samps = int(n_seconds * fs)
    sig = np.random.randn(n_samps)

    # compute FFT
    fc = np.fft.fft(sig)
    f_axis = np.fft.fftfreq(len(sig), 1. / fs)
    # rotate spectrum and invert, zscore to normalize
    fc_rot = spectral.rotate_powerlaw(
        f_axis, fc, exponent / 2., f_rotation=None)
    sig = sp.stats.zscore(np.real(np.fft.ifft(fc_rot)))

    return sig


def sim_filtered_noise(n_seconds, fs, exponent, f_range=(0.5, None), filter_order=None):
    """Simulate colored noise that is highpass or bandpass filtered

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds
    fs : float
        Sampling rate of simulated signal, in Hz
    exponent : float
        Desired power-law exponent - beta in P(f)=f^beta. Negative exponent
        denotes decay (i.e., negative slope in log-log spectrum).
    f_range : 2-element array (lo,hi) or None
        Frequency range of simulated data
        Defaults to (0.5, None), i.e., highpass at 0.5Hz
    filter_order : int
        Order of filter
        Defaults to 3 times the highpass filter cycle length

    Returns
    -------
    noise : np.array
        Filtered noise
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
            print('NOTE: Increased high-pass filter order by 1 in order to be odd')
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
