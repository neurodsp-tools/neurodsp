"""Simulating time series, including oscillations and aperiodic backgrounds."""

import warnings

import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal

from neurodsp import spectral

###################################################################################################
###################################################################################################

def sim_filtered_brown_noise(n_seconds, fs, f_range, filter_order):
    """Simulate a band-pass filtered signal with brown noise.

    Parameters
    ----------
    n_seconds : float
        Length of time of simulated signal, in seconds
    fs : float
        Sampling rate, in Hz
    f_range : 2-element array (lo,hi) or None
        Frequency range of simulated data
            if None: do not filter
    filter_order : int
        Order of filter

    Returns
    -------
    brown_nf : np.array
        Filtered brown noise
    """

    # No filtering
    if f_range is None:

        # No filtering, generate 1/f^2 noise
        brown_n = sim_brown_noise(int(n_seconds * fs))

        return brown_n

    # High pass filtered
    elif f_range[1] is None:

        # Make filter order odd if necessary
        nyq = fs / 2.
        if filter_order % 2 == 0:
            print('NOTE: Increased high-pass filter order by 1 in order to be odd')
            filter_order += 1

        # Generate 1/f^2 noise
        brown_n = sim_brown_noise(int(n_seconds * fs + filter_order * 2))

        # High pass filter
        taps = signal.firwin(filter_order, f_range[0] / nyq, pass_zero=False)
        brown_nf = signal.filtfilt(taps, [1], brown_n)

        return brown_nf[filter_order:-filter_order]

    # Band pass filtered
    else:

        brown_n = sim_brown_noise(int(n_seconds * fs + filter_order * 2))

        # Band pass filter
        nyq = fs / 2.
        taps = signal.firwin(filter_order, np.array(f_range) / nyq, pass_zero=False)
        brown_nf = signal.filtfilt(taps, [1], brown_n)

        return brown_nf[filter_order:-filter_order]


def sim_brown_noise(n_samples):
    """Simulate a brown noise signal (power law distribution 1/f^2).

    Brown noise is simulated by cumulative sum of white noise.

    Parameters
    ----------
    n_samples : int
        Number of samples to simulate

    Returns
    -------
    1d array
        Simulated brown noise signal
    """

    return np.cumsum(np.random.randn(n_samples))


def sim_oscillator(n_samples_cycle, n_cycles, rdsym=.5):
    """Simulate an oscillation.

    Parameters
    ----------
    n_samples_cycle : int
        Number of samples in a single cycle
    n_cycles : int
        Number of cycles to simulate
    rdsym : float
        Rise-decay symmetry of the oscillator, as fraction of the period in the rise time
            =0.5 - symmetric (sine wave)
            <0.5 - shorter rise, longer decay
            >0.5 - longer rise, shorter decay

    Returns
    -------
    oscillator : 1d array
        Oscillating time series
    """

    # Determine number of samples in rise and decay periods
    rise_samples = int(np.round(n_samples_cycle * rdsym))
    decay_samples = n_samples_cycle - rise_samples

    # Make phase array for a single cycle, then repeat it
    pha_one_cycle = np.hstack([np.linspace(
        0, np.pi, decay_samples + 1), np.linspace(-np.pi, 0, rise_samples + 1)[1:-1]])
    phase_t = np.tile(pha_one_cycle, n_cycles)

    # Transform phase into an oscillator
    oscillator = np.cos(phase_t)

    return oscillator


def sim_noisy_oscillator(freq, n_seconds, fs, rdsym=.5, f_hipass_brown=2, SNR=1):
    """Simulate an oscillation embedded in background 1/f.

    Parameters
    ----------
    freq : float
        Oscillator frequency
    n_seconds : float
        Signal duration, in seconds
    fs : float
        Signal sampling rate, in Hz
    f_hipass_brown : float
        Frequency (Hz) at which to high-pass-filter brown noise
    SNR : float
        Ratio of oscillator power to brown noise power
            >1 - oscillator is stronger
            <1 - noise is stronger

    Returns
    -------
    1d array
        Oscillator with brown noise
    """

    # Determine order of highpass filter (3 cycles of f_hipass_brown)
    filter_order = int(3 * fs / f_hipass_brown)
    if filter_order % 2 == 0:
        filter_order += 1

    # Determine length of signal in samples
    n_samples = int(n_seconds * fs)

    # Generate filtered brown noise
    brown = sim_filtered_brown_noise(n_seconds, fs, (f_hipass_brown, None), filter_order)

    # Generate oscillator
    n_samples_cycle = int(fs / freq)
    n_cycles = int(np.ceil(n_samples / n_samples_cycle))
    oscillator = sim_oscillator(n_samples_cycle, n_cycles, rdsym=rdsym)
    oscillator = oscillator[:n_samples]

    # Normalize brown noise power
    oscillator_power = np.mean(oscillator**2)
    brown_power = np.mean(brown**2)
    brown = np.sqrt(brown**2 * oscillator_power / (brown_power * SNR)) * np.sign(brown)

    # Combine oscillator and noise
    output = oscillator + brown

    return output


def sim_bursty_oscillator(freq, n_seconds, fs, rdsym=None, prob_enter_burst=None,
                          prob_leave_burst=None, cycle_features=None,
                          return_cycle_df=False):
    """Simulate a bursty oscillation.

    Parameters
    ----------
    freq : float
        Oscillator frequency, in Hz
    n_seconds : float
        Signal duration, in seconds
    fs : float
        Signal sampling rate, in Hz
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

    # Set default prob_enter_burst and prob_leave_burst and rdsym
    if prob_enter_burst is None:
        prob_enter_burst = .2

    if prob_leave_burst is None:
        prob_leave_burst = .2

    if rdsym is None:
        rdsym = .5

    # Define default parameters for cycle features
    mean_period_samples = int(fs / freq)
    cycle_features_use = {'amp_mean': 1, 'amp_burst_std': .1, 'amp_std': .2,
                          'period_mean': mean_period_samples,
                          'period_burst_std': .1 * mean_period_samples,
                          'period_std': .1 * mean_period_samples,
                          'rdsym_mean': rdsym, 'rdsym_burst_std': .05, 'rdsym_std': .05}

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
        df.drop(df.index[len(df)-1], inplace=True)
        return sig, df
    else:
        return sig


def sim_noisy_bursty_oscillator(freq, n_seconds, fs, rdsym=None, f_hipass_brown=2, SNR=1,
                                prob_enter_burst=None, prob_leave_burst=None,
                                cycle_features=None, return_components=False,
                                return_cycle_df=False):
    """Simulate a bursty oscillation embedded in background 1/f.

    Parameters
    ----------
    freq : float
        Oscillator frequency, in Hz
    n_seconds : float
        Signal duration, in seconds
    fs : float
        Signal sampling rate, in Hz
    rdsym : float
        Rise-decay symmetry of the oscillator as fraction of the period in the rise time
            =0.5 - symmetric (sine wave)
            <0.5 - shorter rise, longer decay
            >0.5 - longer rise, shorter decay
    f_hipass_brown : float
        Frequency, in Hz, at which to high-pass-filter brown noise
    SNR : float
        Ratio of oscillator power to brown noise power
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
        bursty oscillator with brown noise time series
    oscillator : np.array
        bursty oscillator component of signal
    brown : np.array
        brown noise component of signal
    df : pd.DataFrame
        cycle-by-cycle properties of the simulated oscillator
    """

    # Determine order of highpass filter (3 cycles of f_hipass_brown)
    filter_order = int(3 * fs / f_hipass_brown)
    if filter_order % 2 == 0:
        filter_order += 1

    # Generate filtered brown noise
    brown = sim_filtered_brown_noise(n_seconds, fs, (f_hipass_brown, None), filter_order)

    # Generate oscillator
    oscillator, df = sim_bursty_oscillator(freq, n_seconds, fs, rdsym=rdsym,
                                           prob_enter_burst=prob_enter_burst,
                                           prob_leave_burst=prob_leave_burst,
                                           cycle_features=cycle_features,
                                           return_cycle_df=True)

    # Determine samples of burst so that can compute signal power over only those times
    is_osc = np.zeros(len(oscillator), dtype=bool)
    for ind, row in df.iterrows():
        if row['is_cycle']:
            is_osc[row['start_sample']:row['start_sample'] + row['period']] = True

    # Normalize brown noise power
    oscillator_power = np.mean(oscillator[is_osc]**2)
    brown_power = np.mean(brown**2)
    brown = np.sqrt(brown**2 * oscillator_power /
                    (brown_power * SNR)) * np.sign(brown)

    # Combine oscillator and noise
    output = oscillator + brown

    if return_components:
        if return_cycle_df:
            return output, oscillator, brown, df
        return output, oscillator, brown
    else:
        if return_cycle_df:
            return output, df
        return output


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
        Length of simulated signal in seconds
    fs : float
        Sampling rate in Hz
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
        tau_r == tau_d  : alpha (function) synapse
        tau_r = 0       : instantaneous rise, (single) exponential decay
        tau_r!=tau_d!=0 : double-exponential (rise and decay)

    Parameters
    ----------
    t_ker : float
        Length of simulated signal in seconds.
    fs : float
        Sampling rate, in Hz.
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
            warnings.warn('Rise time constant should be shorter than decay time constant.')

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
    T : float
        Length of simulated signal, in seconds
    fs : float
        Sampling rate, in Hz
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
    x : array_like (1D)
        Simulated signal.
    """

    # Simulate an extra bit because the convolution will snip it
    sig = sim_poisson_pop(n_seconds=(n_seconds + t_ker), fs=fs, n_neurons=n_neurons, firing_rate=firing_rate)
    ker = make_synaptic_kernel(t_ker=t_ker, fs=fs, tau_r=tau_r, tau_d=tau_d)

    return np.convolve(sig, ker, 'valid')[:-1]


def sim_OU_process(n_seconds, fs, theta=1., mu=0., sigma=5.):
    """Simulate mean-reverting random walk (Ornstein-Uhlenbeck process)

    Discretized Ornstein-Uhlenbeck process:
        dx = theta*(x-mu)*dt + sigma*dWt, where
    dWt     : increments of Wiener process, i.e. white noise
    theta   : memory scale (higher = faster fluc)
    mu      : mean
    sigma   : std


    Parameters
    ----------
    n_seconds : float
        Length of simulated signal, in seconds
    fs : float
        Sampling rate, in Hz
    theta : float
        Memory scale - larger theta = faster fluctuation
    mu : float
        Mean
    sigma : float
        Standard deviation

    Returns
    -------
    1d array
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

    return x0 * ex + mu * (1. - ex) + sigma * ex * np.cumsum(np.exp(theta * times) * np.sqrt(dt) * ws)


def sim_jittered_oscillator(n_seconds, fs, freq=10., jitter=0, cycle=('gaussian', 0.01)):
    """Simulate a jittered oscillator, as defined by the oscillator frequency,
    the oscillator cycle, and how much (in time) to jitter each period.

    Parameters
    ----------
    T : float
        Simulation length, in seconds
    fs : float
        Sampling frequency, in Hz
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
    1d array
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

    return np.convolve(spks, osc_cycle, 'valid')


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
    1d array
        Simulated oscillation cycle
    """

    if cycle_params[0] == 'gaussian':
        # cycle_params defines std in seconds
        return signal.gaussian(t_ker * fs, cycle_params[1] * fs)

    elif cycle_params[0] == 'exp':
        # cycle_params defines decay time constant in seconds
        return make_synaptic_kernel(t_ker, fs, 0, cycle_params[1])

    elif cycle_params[0] == '2exp':
        # cycle_params defines rise and decay time constant in seconds
        return make_synaptic_kernel(t_ker, fs, cycle_params[1], cycle_params[2])

    else:
        raise ValueError('Did not recognize cycle type.')


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
    1d array
        Time-series with the desired power-law exponent
    """

    n_samps = int(n_seconds * fs)
    sig = np.random.randn(n_samps)

    # compute FFT
    fc = np.fft.fft(sig)
    f_axis = np.fft.fftfreq(len(sig), 1. / fs)

    # rotate spectrum and invert, zscore to normalize
    fc_rot = spectral.rotate_powerlaw(f_axis, fc, exponent/2., f_rotation=None)

    return sp.stats.zscore(np.real(np.fft.ifft(fc_rot)))
