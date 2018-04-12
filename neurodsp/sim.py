"""
sim.py
Simulating oscillators and brown noise
"""

import numpy as np
import pandas as pd
from scipy import signal
import warnings


def sim_filtered_brown_noise(T, Fs, f_range, N):
    """Simulate a band-pass filtered signal with brown noise

    Parameters
    ----------
    T : float
        length of time of simulated oscillation
    Fs : float
        oscillation sampling rate
    f_range : 2-element array (lo,hi)
        frequency range of simulated data
        if None: do not filter
    N : int
        order of filter

    Returns
    -------
    brownNf : np.array
        filtered brown noise
    """

    if f_range is None:
        # Do not filter
        # Generate 1/f^2 noise
        brownN = simbrown(int(T * Fs))
        return brownN
    elif f_range[1] is None:
        # Make filter order odd if necessary
        nyq = Fs / 2.
        if N % 2 == 0:
            print('NOTE: Increased high-pass filter order by 1 in order to be odd')
            N += 1

        # Generate 1/f^2 noise
        brownN = sim_brown_noise(int(T * Fs + N * 2))

        # High pass filter
        taps = signal.firwin(N, f_range[0] / nyq, pass_zero=False)
        brownNf = signal.filtfilt(taps, [1], brownN)
        return brownNf[N:-N]

    else:
        # Bandpass filter
        # Generate 1/f^2 noise
        brownN = simbrown(int(T * Fs + N * 2))
        # Filter
        nyq = Fs / 2.
        taps = signal.firwin(N, np.array(f_range) / nyq, pass_zero=False)
        brownNf = signal.filtfilt(taps, [1], brownN)
        return brownNf[N:-N]


def sim_brown_noise(N):
    """Simulate a brown noise signal (power law distribution 1/f^2)
    with N samples by cumulative sum of white noise"""
    return np.cumsum(np.random.randn(N))


def sim_oscillator(N_samples_cycle, N_cycles, rdsym=.5):
    """Simulate a band-pass filtered signal with 1/f^2
    Input suggestions: f_range=(2,None), Fs=1000, N=1001

    Parameters
    ----------
    N_samples_cycle : int
        Number of samples in a single cycle
    N_cycles : int
        Number of cycles to simulate
    rdsym : float
        rise-decay symmetry of the oscillator;
        fraction of the period in the rise time;
        =0.5 - symmetric (sine wave)
        <0.5 - shorter rise, longer decay
        >0.5 - longer rise, shorter decay

    Returns
    -------
    oscillator : np.array
        oscillating time series
    """
    # Determine number of samples in rise and decay periods
    rise_samples = int(np.round(N_samples_cycle * rdsym))
    decay_samples = N_samples_cycle - rise_samples

    # Make phase array for a single cycle, then repeat it
    pha_one_cycle = np.hstack([np.linspace(
        0, np.pi, decay_samples + 1), np.linspace(-np.pi, 0, rise_samples + 1)[1:-1]])
    phase_t = np.tile(pha_one_cycle, N_cycles)

    # Transform phase into an oscillator
    oscillator = np.cos(phase_t)
    return oscillator


def sim_noisy_oscillator(freq, T, Fs, rdsym=.5, f_hipass_brown=2, SNR=1):
    """Simulate a band-pass filtered signal with 1/f^2
    Input suggestions: f_range=(2,None), Fs=1000, N=1001

    Parameters
    ----------
    freq : float
        oscillator frequency
    T : float
        signal duration (seconds)
    Fs : float
        signal sampling rate
    f_hipass_brown : float
        frequency (Hz) at which to high-pass-filter
        brown noise
    SNR : float
        ratio of oscillator power to brown noise power
        >1 - oscillator is stronger
        <1 - noise is stronger

    Returns
    -------
    signal : np.array
        oscillator with brown noise
    """

    # Determine order of highpass filter (3 cycles of f_hipass_brown)
    N = int(3 * Fs / f_hipass_brown)
    if N % 2 == 0:
        N += 1

    # Determine length of signal in samples
    N_samples = int(T * Fs)

    # Generate filtered brown noise
    brown = sim_filtered_brown_noise(T, Fs, (f_hipass_brown, None), N)

    # Generate oscillator
    N_samples_cycle = int(Fs / freq)
    N_cycles = int(np.ceil(N_samples / N_samples_cycle))
    oscillator = sim_oscillator(N_samples_cycle, N_cycles, rdsym=rdsym)
    oscillator = oscillator[:N_samples]

    # Normalize brown noise power
    oscillator_power = np.mean(oscillator**2)
    brown_power = np.mean(brown**2)
    brown = np.sqrt(brown**2 * oscillator_power /
                    (brown_power * SNR)) * np.sign(brown)
    # Combine oscillator and noise
    signal = oscillator + brown
    return signal


def sim_bursty_oscillator(freq, T, Fs, rdsym=None, prob_enter_burst=None,
                          prob_leave_burst=None, cycle_features=None,
                          return_cycle_df=False):
    """Simulate a band-pass filtered signal with 1/f^2
    Input suggestions: f_range=(2,None), Fs=1000, N=1001

    Parameters
    ----------
    freq : float
        oscillator frequency
    T : float
        signal duration (seconds)
    Fs : float
        signal sampling rate
    rdsym : float
        rise-decay symmetry of the oscillator;
        fraction of the period in the rise time;
        =0.5 - symmetric (sine wave)
        <0.5 - shorter rise, longer decay
        >0.5 - longer rise, shorter decay
    prob_enter_burst : float
        probability of a cycle being oscillating given
        the last cycle is not oscillating
    prob_leave_burst : float
        probability of a cycle not being oscillating
        given the last cycle is oscillating
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
    signal : np.array
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
    mean_period_samples = int(Fs / freq)
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
    N_samples = T * Fs
    N_cycles_overestimate = int(np.ceil(N_samples / mean_period_samples * 2))

    # Simulate if a series of cycles are oscillating or not oscillating
    is_oscillating = [False]
    N_cycles_current = 1
    while N_cycles_current < N_cycles_overestimate:
        rand_num = np.random.rand()
        if is_oscillating[-1]:
            is_oscillating.append(rand_num > prob_leave_burst)
        else:
            is_oscillating.append(rand_num < prob_enter_burst)
        N_cycles_current += 1

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
    df = df[df['start_sample'] < N_samples]

    # Shorten df to only cycles that are included in the data

    # Simulate time series for each cycle
    x = np.array([])
    last_cycle_oscillating = False
    for i, row in df.iterrows():
        if row['is_cycle'] is False:
            # If last cycle was oscillating, add a decay to 0 then 0s
            if last_cycle_oscillating:
                decay_pha = np.linspace(0, np.pi / 2, int(row['period'] / 4))
                decay_t = np.cos(decay_pha) * x[-1]
                x = np.append(x, decay_t)

                cycle_t = np.zeros(row['period'] - int(row['period'] / 4))
                x = np.append(x, cycle_t)
            else:
                # Add a blank cycle
                cycle_t = np.zeros(row['period'])
                x = np.append(x, cycle_t)
            last_cycle_oscillating = False
        else:
            # If last cycle was oscillating, add a decay to 0
            if not last_cycle_oscillating:
                rise_pha = np.linspace(-np.pi / 2, 0,
                                       int(row['period'] / 4))[1:]
                rise_t = np.cos(rise_pha) * row['amp']
                x[-len(rise_t):] = rise_t

            # Add a cycle with rdsym
            rise_samples = int(np.round(row['period'] * row['rdsym']))
            decay_samples = row['period'] - rise_samples
            pha_t = np.hstack([np.linspace(0, np.pi, decay_samples + 1)[1:],
                               np.linspace(-np.pi, 0, rise_samples + 1)[1:]])
            cycle_t = np.cos(pha_t)

            # Adjust decay if the last cycle was oscillating
            if last_cycle_oscillating:
                scaling = (row['amp'] + x[-1]) / 2
                offset = (x[-1] - row['amp']) / 2
                cycle_t[:decay_samples] = cycle_t[:decay_samples] * \
                    scaling + offset
                cycle_t[decay_samples:] = cycle_t[decay_samples:] * row['amp']
            else:
                cycle_t = cycle_t * row['amp']
            x = np.append(x, cycle_t)
            last_cycle_oscillating = True
    x = x[:N_samples]

    if return_cycle_df:
        return x, df
    else:
        return x


def sim_noisy_bursty_oscillator(freq, T, Fs, rdsym=None, f_hipass_brown=2, SNR=1,
                                prob_enter_burst=None, prob_leave_burst=None,
                                cycle_features=None, return_components=False,
                                return_cycle_df=False):
    """Simulate a band-pass filtered signal with 1/f^2
    Input suggestions: f_range=(2,None), Fs=1000, N=1001

    Parameters
    ----------
    freq : float
        oscillator frequency
    T : float
        signal duration (seconds)
    Fs : float
        signal sampling rate
    rdsym : float
        rise-decay symmetry of the oscillator;
        fraction of the period in the rise time;
        =0.5 - symmetric (sine wave)
        <0.5 - shorter rise, longer decay
        >0.5 - longer rise, shorter decay
    f_hipass_brown : float
        frequency (Hz) at which to high-pass-filter
        brown noise
    SNR : float
        ratio of oscillator power to brown noise power
        >1 - oscillator is stronger
        <1 - noise is stronger
    prob_enter_burst : float
        probability of a cycle being oscillating given
        the last cycle is not oscillating
    prob_leave_burst : float
        probability of a cycle not being oscillating
        given the last cycle is oscillating
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
        if True, return the oscillator and noise separate,
        in addition to the signal
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
    N = int(3 * Fs / f_hipass_brown)
    if N % 2 == 0:
        N += 1

    # Determine length of signal in samples
    N_samples = int(T * Fs)

    # Generate filtered brown noise
    brown = sim_filtered_brown_noise(T, Fs, (f_hipass_brown, None), N)

    # Generate oscillator
    oscillator, df = sim_bursty_oscillator(freq, T, Fs, rdsym=rdsym,
                                           prob_enter_burst=prob_enter_burst,
                                           prob_leave_burst=prob_leave_burst,
                                           cycle_features=cycle_features,
                                           return_cycle_df=True)

    # Determine samples of burst so that can compute signal power over only those times
    is_osc = np.zeros(len(oscillator), dtype=bool)
    for i, row in df.iterrows():
        if row['is_cycle']:
            is_osc[row['start_sample']:row['start_sample'] + row['period']] = True

    # Normalize brown noise power
    oscillator_power = np.mean(oscillator[is_osc]**2)
    brown_power = np.mean(brown**2)
    brown = np.sqrt(brown**2 * oscillator_power /
                    (brown_power * SNR)) * np.sign(brown)
    # Combine oscillator and noise
    signal = oscillator + brown

    if return_components:
        if return_cycle_df:
            return signal, oscillator, brown, df
        return signal, oscillator, brown
    else:
        if return_cycle_df:
            return signal, df
        return signal


def sim_poisson_pop(T, Fs, N_neurons, FR):
    """Simulates a poisson population. It is essentially white noise, but satisfies
    the Poisson property, i.e. mean(X) = var(X).

    The lambda parameter of the Poisson process (total rate) is determined as
    firing rate * number of neurons, i.e. summation of poisson processes is still
    a poisson processes.

    Note that the Gaussian approximation for a sum of Poisson processes is only
    a good approximation for large lambdas.

    Parameters
    ----------
    T : float
        Length of simulated signal in seconds.
    Fs : float
        Sampling rate in Hz.
    N_neurons : int
        Number of neurons in the simulated population.
    FR : type
        Firing rate of individual neurons in the population.

    Returns
    -------
    x : array (1-D)
        Simulated population activity.

    """
    L = int(T * Fs)
    # poisson population rate signal scales with # of neurons and individual rate
    lam = N_neurons * FR

    # variance is equal to the mean
    x = np.random.normal(loc=lam, scale=lam**0.5, size=L)

    # enforce that X is non-negative in cases of low FR
    x[np.where(x < 0.)] = 0.
    return x


def make_synaptic_kernel(T_ker, Fs, tauR, tauD):
    """
    Creates synaptic kernels that with specified time constants.

    3 types of kernels are available, based on combinations of time constants:
                tauR == tauD : alpha (function) synapse
        tauR = 0     : instantaneous rise, (single) exponential decay
        tauR!=tauD!=0: double-exponential (rise and decay)

    Parameters
    ----------
    T_ker : float
        Length of simulated signal in seconds.
    Fs : float (Hz)
        Sampling rate.
    tauR : float, seconds
            Rise time of synaptic kernel.
    tauD : fload, seconds
                Decay time of synaptic kernel.

    Returns
    -------
    kernel : array_like
                Computed synaptic kernel with length equal to t

    """
    t = np.arange(0, T_ker, 1 / Fs)
    if tauR == 0:
        # single exponential synapse
        kernel = np.exp(-t / tauD)
        ktype = 'single exponential'

    elif tauR == tauD:
        # alpha synapse
        # I(t) = t/tau * exp(-t/tau)
        kernel = (t / tauR) * np.exp(-t / tauR)
        ktype = 'alpha'

    else:
        # double exponential synapse of the form:
        # I(t)=(tauR/(tauR-tauD))*(exp(-t/tauD)-exp(-t/tauR))
        if tauR > tauD:
            warnings.warn(
                'Rise time constant should be shorter than decay time constant.')
        kernel = (np.exp(-t / tauD) - np.exp(-t / tauR))
        ktype = 'double exponential'

    kernel = kernel / np.sum(kernel)  # normalize the integral to 1
    return kernel


def sim_synaptic_noise(T, Fs, N_neurons=1000, FR=2, T_ker=1., tauR=0, tauD=0.01):
    """ Simulate a neural signal with 1/f characteristics beyond a knee frequency.
    The resulting signal is most similar to unsigned intracellular current or
    conductance change.

    Parameters
    ----------
    T : float
        Length of simulated signal in seconds.
    Fs : float
        Sampling rate in Hz.
    N_neurons : int
        Number of neurons in the simulated population.
    FR : type
        Firing rate of individual neurons in the population.
    T_ker : float
        Length of simulated kernel in seconds. Usually 1 second will suffice.
    tauR : float, seconds
            Rise time of synaptic kernel.
    tauD : fload, seconds
                Decay time of synaptic kernel.

    Returns
    -------
    x : array_like (1D)
        Simulated signal.

    """
    # simulate an extra bit because the convolution will snip it
    x = sim_poisson_pop(T=(T + T_ker), Fs=Fs, N_neurons=N_neurons, FR=FR)
    ker = make_synaptic_kernel(T_ker=T_ker, Fs=Fs, tauR=tauR, tauD=tauD)
    return np.convolve(x, ker, 'valid')[:-1]


def sim_OU_process(T, Fs, theta=1., mu=0., sigma=5.):
    """Simulate mean-reverting random walk (Ornstein-Uhlenbeck process)

    Discretized Ornstein-Uhlenbeck process:
        dx = theta*(x-mu)*dt + sigma*dWt, where
    dWt: increments of Wiener process, i.e. white noise
    theta: memory scale (higher = faster fluc)
    mu: mean
    sigma: std

    see: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process#Solution
    for integral solution

    Parameters
    ----------
    T : float
        Length of simulated signal in seconds.
    Fs : float
        Sampling rate in Hz.
    theta : float, default = 1.
        Memory scale (larger theta = faster fluctuation).
    mu : float, default = 0.
        Mean.
    sigma : float, default = 5.
        Standard deviation.

    Returns
    -------
    x : array (1-D)
        Simulated signal.

    """

    t = np.arange(0, T, 1 / Fs)
    x0 = mu
    dt = t[1] - t[0]
    Ws = np.random.normal(size=len(t))
    ex = np.exp(-theta * t)
    Ws[0] = 0.
    return x0 * ex + mu * (1. - ex) + sigma * ex * np.cumsum(np.exp(theta * t) * np.sqrt(dt) * Ws)


def sim_jittered_oscillator(T, Fs, freq=10., jitter=0, cycle=('gaussian', 0.01)):
    """ Simulated a jittered oscillator, as defined by the oscillator frequency,
    the oscillator cycle, and how much (in time) to jitter each period.

    Parameters
    ----------
    T : float (seconds)
        Simulation length.
    Fs : float (Hz)
        Sampling frequency.
    freq : float (Hz)
        Frequency of simulated oscillator.
    jitter : float (seconds)
        Maximum jitter of oscillation period.
    cycle : tuple or array (1D)
        Oscillation cycle used in the simulation.
        If array, it's used directly.
        If tuple, it is generated based on given parameters.
        Possible values:
        ('gaussian', std): gaussian cycle, standard deviation in seconds
        ('exp', decay time): exponential decay, decay time constant in seconds
        ('2exp', rise time, decay time): exponential rise and decay

    Returns
    -------
    x : array (1D)
        Simulated oscillation with jitter.

    """

    # if cycle is a tuple, generate the window with given params, and if
    # cycle is an array, just use it to do the convolution
    if type(cycle) is tuple:
        # defaults to 1 second window for a cycle, which is more than enough
        # if interested in longer period oscillations, just pass in premade cycle
        osc_cycle = make_osc_cycle(1, Fs, cycle)
    else:
        osc_cycle = cycle

    # binary "spike-train" of when each cycle should occur
    spks = np.zeros(int(T * Fs + len(osc_cycle)) - 1)
    osc_period = int(Fs / freq)
    # generate oscillation "event" indices
    spk_indices = np.arange(osc_period, len(spks), osc_period)

    if jitter != 0:
        # add jitter to "spike" indices
        spk_indices = spk_indices + \
            np.random.randint(low=-int(Fs * jitter),
                              high=int(Fs * jitter), size=len(spk_indices))

    spks[spk_indices] = 1

    return np.convolve(spks, osc_cycle, 'valid')


def make_osc_cycle(T_ker, Fs, cycle_params):
    """ Make 1 cycle of oscillation.

    Parameters
    ----------
    T_ker : float
        Length of cycle window in seconds.
        Note that this is NOT the period of the cycle, but the length of the
        returned array that contains the cycle, which can be (and usually is)
        much shorter.
    Fs : float
        Sampling frequency of the cycle simulation.
    cycle_params : tuple
        Defines the parameters for the oscillation cycle.
        Possible values:
        ('gaussian', std): gaussian cycle, standard deviation in seconds
        ('exp', decay time): exponential decay, decay time constant in seconds
        ('2exp', rise time, decay time): exponential rise and decay

    Returns
    -------
    x: array_like (1D)
        Simulated oscillation cycle.

    """
    if cycle_params[0] is 'gaussian':
        # cycle_params defines std in seconds
        return signal.gaussian(T_ker * Fs, cycle_params[1] * Fs)
    elif cycle_params[0] is 'exp':
        # cycle_params defines decay time constant in seconds
        return make_synaptic_kernel(T_ker, Fs, 0, cycle_params[1])
    elif cycle_params[0] is '2exp':
        # cycle_params defines rise and decay time constant in seconds
        return make_synaptic_kernel(T_ker, Fs, cycle_params[1], cycle_params[2])
    else:
        # Is this the proper way to handle errors???
        print('Did not recognize cycle type.')
        return None
