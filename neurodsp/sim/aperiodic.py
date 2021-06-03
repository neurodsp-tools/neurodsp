"""Simulating time series, with aperiodic activity."""

import numpy as np
from scipy.stats import zscore
from scipy.linalg import toeplitz, cholesky

from neurodsp.filt import filter_signal, infer_passtype
from neurodsp.filt.fir import compute_filter_length
from neurodsp.filt.checks import check_filter_definition
from neurodsp.utils import remove_nans
from neurodsp.utils.data import create_times, compute_nsamples
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
    sig = np.random.normal(loc=lam, scale=lam**0.5, size=compute_nsamples(n_seconds, fs))

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
    sig = np.convolve(sig, ker, 'valid')[:compute_nsamples(n_seconds, fs)]

    return sig


@normalize
def sim_knee(n_seconds, fs, chi1, chi2, knee):
    """Returns a time series whose power spectrum follows the Lorentzian equation:

    P(f) = 1 / (f**chi1 * (f**chi2 + knee))

    using a sum of sinusoids.

    Parameters
    -----------
    n_seconds: float
        Number of seconds elapsed in the time series.
    fs: float
        Sampling rate.
    chi1: float
        Power law exponent before the knee.
    chi2: float
        Power law exponent added to chi1 after the knee.
    knee: float
        Location of the knee in hz.

    Returns
    -------
    sig: 1d array
        Time series with desired power spectrum.

    Notes
    -----
    The slope of the log power spectrum before the knee is chi1 whereas after the knee it is chi2,
    but only when the sign of chi1 and chi2 are the same.

    Examples
    --------
    Simulate a time series with (chi1, chi2, knee) = (-1, -2, 100)

    >> sim_knee(n_seconds=10, fs=10**3, chi1=-1, chi2=-2, knee=100)
    """

    times = create_times(n_seconds, fs)
    n_samples = compute_nsamples(n_seconds, fs)

    # Create the range of frequencies that appear in the power spectrum since these
    #   will be the frequencies in the cosines we sum below
    freqs = np.linspace(0, fs/2, num=int(n_samples//2 + 1), endpoint=True)

    # Drop the DC component
    freqs = freqs[1:]

    # Map the frequencies under the (square root) Lorentzian
    #   This will give us the amplitude coefficients for the sinusoids
    cosine_coeffs = np.array([np.sqrt(1/(f**-chi1*(f**(-chi2-chi1) + knee))) for f in freqs])

    # Add sinusoids with a random phase shift
    sig = np.sum(
        np.array(
            [cosine_coeffs[ell]*np.cos(2*np.pi*f*times + 2*np.pi*np.random.rand())
             for ell, f in enumerate(freqs)]
        ),
        axis=0
    )

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

    # Compute the number of samples for the simulated time series
    n_samples = compute_nsamples(n_seconds, fs)

    # Get the number of samples to simulate for the signal
    #   If signal is to be filtered, with FIR, add extra to compensate for edges
    if f_range and filter_kwargs.get('filter_type', None) != 'iir':

        pass_type = infer_passtype(f_range)
        filt_len = compute_filter_length(fs, pass_type,
                                         *check_filter_definition(pass_type, f_range),
                                         n_seconds=filter_kwargs.get('n_seconds', None),
                                         n_cycles=filter_kwargs.get('n_cycles', 3))

        n_samples += filt_len + 1

    # Simulate the powerlaw data
    sig = _create_powerlaw(n_samples, fs, exponent)

    if f_range is not None:
        sig = filter_signal(sig, fs, infer_passtype(f_range), f_range,
                            remove_edges=True, **filter_kwargs)
        # Drop the edges, that were compensated for, if not using FIR filter
        if not filter_kwargs.get('filter_type', None) == 'iir':
            sig, _ = remove_nans(sig)

    return sig

@normalize
def sim_frac_gaussian_noise(n_seconds, fs, chi=0, hurst=None):
    """Simulate a fractional gaussian noise time series with a specified
    power law exponent, or alternatively with a specified Hurst parameter.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    chi: float, optional, default: 0
        Desired power law exponent for the spectrogram. Must be in the range (-1, 1).
    hurst : float, optional, default: None
        Desired Hurst parameter. Overwrites chi if defined. Must be in the range (0, 1).

    Returns
    -------
    sig: 1d array
        Fractional gaussian noise time series with  the desired power law decay in
        the spectrogram, or the desired Hurst parameter.

    Notes
    -----
    The Hurst parameter is not the Hurst exponent as defined in rescaled range analysis.
    The Hurst parameter is defined for self-similar processes such that Y(at) = a^H Y(t)
    for all a > 0, where this equality holds in distribution.

    The relationship between the power law exponent chi and the Hurst parameter
    for fractional gaussian noise is chi = 2 * hurst - 1.

    References
    ----------
    For more information, consult Eke, A., et al. "Fractal characterization of
    complexity in temporal physiological signals." Physiological measurement 23.1 (2002): R1.

    Examples
    --------
    Simulate fractional gaussian noise with a power law decay of 0, or
    equivalently with a Hurst parameter of 0.5 (white noise):

    >>> sig = sim_fgn(n_seconds=1, fs=500)
    """

    if hurst is not None:
        # Check that hurst is defined in the range (0, 1)
        if (hurst <= 0 or hurst >= 1):
            raise ValueError("Hurst parameter must be chosen from the open interval (0,1).")
    else:
        # Check that chi is defined in the range (-1, 1)
        if chi <= -1 or chi >= 1:
            raise ValueError("Chi must be chosen from the open interval (-1, 1).")

        # Infer the hurst parameter from chi
        hurst = (-chi + 1.)/2

    # Construct the autocovariance function
    n_samples = int(n_seconds * fs)
    gamma = np.arange(0, n_samples)
    def autocov(hurst):
        return lambda k : 0.5*(np.abs(k-1)**(2 * hurst) - 2*k**(2*hurst) + (k+1)**(2*hurst))
    gamma = np.apply_along_axis(autocov(hurst), 0, gamma)

    # Build the autocovariance matrix
    #   Use the Cholesky factor to transform white noise to get the desired time series
    autocov_matrix = toeplitz(gamma)
    cholesky_factor = cholesky(autocov_matrix, lower=True)

    white_noise = np.random.randn(n_samples)
    return cholesky_factor @ white_noise

@normalize
def sim_frac_brownian_motion(n_seconds, fs, chi=-2, hurst=None):
    """Simulate a fractional brownian motion with a specified power law exponent,
    or alternatively with a specified Hurst parameter.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Sampling rate of simulated signal, in Hz.
    chi: float, optional, default: -2
        Desired power law exponent for the spectrogram. Must be in the range (-3, -1).
    hurst : float, optional, default: None
        Desired Hurst parameter. Overwrites chi if defined. Must be in the range (0, 1).

    Returns
    -------
    sig: 1d array
        Fractional brownian motion with the desired the desired power law decay
         in the spectrogram, or alternatively with the specified Hurst parameter.

    Notes
    -----
    The Hurst parameter is not the Hurst exponent in general. The Hurst parameter
    is defined for self-similar processes such that Y(at) = a^H Y(t) for all a > 0,
    where this equality holds in distribution.

    The relationship between the power law exponent chi and the Hurst parameter
    for fractional brownian motion is chi = 2 * hurst + 1

    References
    ----------
    For more information, consult Eke, A., et al. "Fractal characterization of
    complexity in temporal physiological signals." Physiological measurement 23.1 (2002): R1.

    Examples
    --------
    Simulate fractional brownian motion with a power law exponent of -2, or
    equivalently with a Hurst parameter of 0.5 (brown noise):

    >>> sig = sim_fbm(n_seconds=1, fs=500)
    """

    if hurst is not None:
        # Check that hurst is defined in (0, 1)
        if hurst <= 0 or hurst >= 1:
            raise ValueError("Hurst parameter must be chosen from the open interval (0,1).")
    else:
        # Check that chi is defined in (1, 3)
        if chi <= -3 or chi >= -1:
            raise ValueError("Chi must be chosen from the open interval (-3, -1).")

        # Infer the hurst parameter from chi
        hurst = (-chi - 1.)/2

    # Fractional brownian motion is the cumulative sum of fractional gaussian noise
    fgn = sim_frac_gaussian_noise(n_seconds, fs, hurst=hurst)
    return np.cumsum(fgn)

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
