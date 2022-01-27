"""Simulating time series, with combinations of periodic, aperiodic and transient components."""

from itertools import repeat

import numpy as np
from scipy.linalg import norm

from neurodsp.sim.info import get_sim_func
from neurodsp.utils.decorators import normalize
from neurodsp.utils.data import create_times

###################################################################################################
###################################################################################################

@normalize
def sim_combined(n_seconds, fs, components, component_variances=1):
    """Simulate a signal by combining multiple component signals.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    components : dictionary
        A dictionary of simulation functions to run, with their desired parameters.
    component_variances : list of float or 1, optional, default: 1
        Variance to simulate with for each component of the signal.
        If 1, each component signal is simulated with unit variance.

    Returns
    -------
    sig : 1d array
        Simulated combined signal.

    Examples
    --------
    Simulate a combined signal with an aperiodic and periodic component:

    >>> sim_components = {'sim_powerlaw': {'exponent' : -2}, 'sim_oscillation': {'freq' : 10}}
    >>> sig = sim_combined(n_seconds=1, fs=500, components=sim_components)

    Simulate a combined signal with multiple periodic components:

    >>> sim_components = {'sim_powerlaw': {'exponent' : -2},
    ...                   'sim_oscillation': [{'freq' : 10}, {'freq' : 20}]}
    >>> sig = sim_combined(n_seconds=1, fs=500, components=sim_components)

    Simulate a combined signal with unequal variance for the different components:

    >>> sig = sim_combined(n_seconds=1, fs=500,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation': {'freq' : 10}},
    ...                    component_variances=[0.25, 0.75])
    """

    # Check how simulation components are specified, in terms of number of parameter sets
    n_sims = sum([1 if isinstance(params, dict) else len(params) \
        for params in components.values()])

    # Check that the variance definition matches the number of components specified
    if not (component_variances == 1 or len(component_variances) == n_sims):
        raise ValueError('Signal components and variances lengths do not match.')

    # Collect the sim function to use, and repeat variance if is single number
    components = {(get_sim_func(name) if isinstance(name, str) else name) : params \
                   for name, params in components.items()}
    variances = repeat(component_variances) if isinstance(component_variances, (int, float)) \
        else iter(component_variances)

    # Simulate each component of the signal
    sig_components = []
    for func, params in components.items():

        # If list, params should be a list of separate parameters for each function call
        if isinstance(params, list):
            sig_components.extend([func(n_seconds=n_seconds, fs=fs, **cur_params,
                                        variance=next(variances)) for cur_params in params])

        # Otherwise, params should be a dictionary of parameters for single call
        else:
            sig_components.append(func(n_seconds=n_seconds, fs=fs, **params,
                                       variance=next(variances)))

    # Combine total signal across all simulated components
    sig = np.sum(sig_components, axis=0)

    return sig


@normalize
def sim_peak_oscillation(sig_ap, fs, freq, bw, height):
    """Simulate a signal with an aperiodic component and a specific oscillation peak.

    Parameters
    ----------
    sig_ap : 1d array
        The timeseries of the aperiodic component.
    fs : float
        Sampling rate of ``sig_ap``.
    freq : float
        Central frequency for the gaussian peak in Hz.
    bw : float
        Bandwidth, or standard deviation, of gaussian peak in Hz.
    height : float
        Relative height of the gaussian peak at the central frequency ``freq``.
        Units of log10(power), over the aperiodic component.

    Returns
    -------
    sig : 1d array
        Time series with desired power spectrum.

    Notes
    -----
    - This function creates a time series whose power spectrum consists of an aperiodic component
    and a gaussian peak at ``freq`` with standard deviation ``bw`` and relative ``height``.
    - The periodic component of the signal will be sinusoidal.

    Examples
    --------
    Simulate a signal with aperiodic exponent of -2 & oscillation central frequency of 20 Hz:

    >>> from neurodsp.sim import sim_powerlaw
    >>> fs = 500
    >>> sig_ap = sim_powerlaw(n_seconds=10, fs=fs, exponent=-2.0)
    >>> sig = sim_peak_oscillation(sig_ap, fs=fs, freq=20, bw=5, height=7)
    """

    sig_len = len(sig_ap)
    times = create_times(sig_len / fs, fs)

    # Construct the aperiodic component and compute its Fourier transform
    # Only use the first half of the frequencies from the FFT since the signal is real
    sig_ap_hat = np.fft.fft(sig_ap)[0:(sig_len // 2 + 1)]

    # Create the range of frequencies that appear in the power spectrum since these
    # will be the frequencies in the cosines we sum below
    freqs = np.linspace(0, fs / 2, num=sig_len // 2 + 1, endpoint=True)

    # Define sub-function for computing the relative height above the aperiodic power spectrum
    def _hght(f_val):
        return height * np.exp(-(f_val - freq) ** 2 / (2 * bw ** 2))

    # Define sub-function for computing the sum of squares of the cosines
    def _cosn(f_val):
        return norm(np.cos(2 * np.pi * f_val * times), 2) ** 2

    # Define sub-function for computing the amplitude coefficients
    def _coef(f_val, fft):
        return (-np.real(fft) + np.sqrt(np.real(fft) ** 2 + (10 ** _hght(f_val) - 1) * \
            np.abs(fft) ** 2)) / _cosn(f_val)

    # Define & vectorize sub-function to create the set of cosines for the periodic component
    #   Cosines are created with their respective coefficients & a random phase shift
    def _pers(f_val, fft):
        return _coef(f_val, fft) * np.cos(2 * np.pi * f_val * times + 2 * np.pi * np.random.rand())
    vect_pers = np.vectorize(_pers, signature='(),()->(n)')

    # Create the set of cosines, defined from the frequency and FFT values
    sines = vect_pers(freqs, sig_ap_hat)

    # Create the periodic component by summing across the simulated cosines
    sig_periodic = np.sum(sines, axis=0)

    # Create the combined signal by summing periodic & aperiodic
    sig = sig_ap + sig_periodic

    return sig
