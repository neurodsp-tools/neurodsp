"""Simulating time series, with combinations of periodic, aperiodic and transient components."""

import numpy as np

from neurodsp.sim.periodic import sim_oscillation, sim_bursty_oscillation
from neurodsp.sim.aperiodic import _return_noise_sim

###################################################################################################
###################################################################################################

def sim_noisy_oscillation(n_seconds, fs, freq, noise_generator='synaptic',
                          noise_args={}, rdsym=.5, ratio_osc_var=1):
    """Simulate an oscillation embedded in background 1/f noise.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    freq : float
        Oscillation frequency.
    noise_generator: str or numpy.ndarray, optional, default='synaptic'
        Noise model, can be one of the simulators in neurodsp.sim specificed as a string, or a
        custom array with the same number of samples as the oscillation (n_seconds*fs).

        Possible models (see respective documentation):

        - 'synaptic' or 'lorentzian': sim.sim_synaptic_noise()
          Defaults: n_neurons=1000, firing_rate=2, tau_r=0, tau_d=0.01
        - 'filtered_powerlaw': sim.sim_filtered_noise()
          Defaults: exponent=-2., f_range=(0.5, None), filter_order=None
        - 'powerlaw': sim.sim_variable_powerlaw()
          Defaults: exponent=-2.0
        - 'ou_process': sim.sim_ou_process()
          Defaults: theta=1., mu=0., sigma=5.
    noise_args: dict('argname':argval, ...)
        Function arguments for the neurodsp.sim noise generaters. See API for arg names.
        All args are optional, defaults for each noise generator are listed above.
    rdsym : float
        Rise-decay symmetry of the oscillation, as fraction of the period in the rise time:

        - = 0.5 - symmetric (i.e., sine wave, default)
        - < 0.5 - shorter rise, longer decay
        - > 0.5 - longer rise, shorter decay
    ratio_osc_var : float, optional, default=1
        Ratio of oscillation variance to noise variance.
        If >1 - oscillation is stronger, if <1 - noise is stronger.

    Returns
    -------
    osc: 1d array
        Oscillation with noise.
    """

    # Generate & demean noise
    noise = _return_noise_sim(n_seconds, fs, noise_generator, noise_args)
    noise = noise - noise.mean()

    # Generate oscillation
    oscillation = sim_oscillation(n_seconds, fs, freq, rdsym=rdsym)

    # Normalize & combine signal
    oscillation, noise = normalize_by_variance(oscillation, noise, ratio_osc_var)
    osc = oscillation + noise

    return osc


def sim_noisy_bursty_oscillation(n_seconds, fs, freq, noise_generator='synaptic', noise_args={},
                                 rdsym=.5, ratio_osc_var=1, prob_enter_burst=.2, prob_leave_burst=.2,
                                 cycle_features=None, return_components=False, return_cycle_df=False):
    """Simulate a bursty oscillation embedded in background 1/f noise.

    Parameters
    ----------
    n_seconds : float
        Simulation time, in seconds
    fs : float
        Sampling rate of simulated signal, in Hz
    freq : float
        Oscillation frequency, in Hz
    noise_generator: str or numpy.ndarray, optional, default: 'synaptic'
        Noise model, can be one of the simulators in neurodsp.sim specificed as a string, or a custom
        numpy.ndarray with the same number of samples as the oscillation (n_seconds*fs).

        Possible models (see respective documentation):

        - 'synaptic' or 'lorentzian': sim.sim_synaptic_noise()
          Defaults: n_neurons=1000, firing_rate=2, tau_r=0, tau_d=0.01
        - 'filtered_powerlaw': sim.sim_filtered_noise()
          Defaults: exponent=-2., f_range=(0.5, None), filter_order=None
        - 'powerlaw': sim.sim_variable_powerlaw()
          Defaults: exponent=-2.0
        - 'ou_process': sim.sim_ou_process()
          Defaults: theta=1., mu=0., sigma=5.
    noise_args: dict('argname':argval, ...)
        Function arguments for the neurodsp.sim noise generaters. See API for arg names.
        All args are optional, defaults for each noise generator are listed above.
    rdsym : float
        Rise-decay symmetry of the oscillation as fraction of the period in the rise time

        - = 0.5: symmetric (sine wave)
        - < 0.5: shorter rise, longer decay
        - > 0.5: longer rise, shorter decay
    ratio_osc_var : float
        Ratio of oscillation power to noise power.
        If >1 - oscillation is stronger, if <1 - noise is stronger.
    prob_enter_burst : float
        Probability of a cycle being oscillating given the last cycle is not oscillating.
    prob_leave_burst : float
        Probability of a cycle not being oscillating given the last cycle is oscillating.
    cycle_features : dict
        Specifies the mean and standard deviations (within and across bursts) of each cycle's
        amplitude, period, and rise-decay symmetry.
        This can include a complete or incomplete set (using defaults) of the following keys:

        * amp_mean: mean cycle amplitude
        * amp_std: standard deviation of cycle amplitude
        * amp_burst_std: standard deviation of mean amplitude for each burst
        * period_mean: mean period (computed from `freq`)
        * period_std: standard deviation of period (samples)
        * period_burst_std: standard deviation of mean period for each burst
        * rdsym_mean: mean rise-decay symmetry
        * rdsym_std: standard deviation of rdsym
        * rdsym_burst_std: standard deviation of mean rdsym for each burst
    return_components: bool
        Whether to return the oscillation and noise separately, in addition to the signal.
    return_cycle_df : bool
        If True, return the dataframe that contains the simulation parameters for each cycle.
        This may be useful for computing power, for example, as the power of the oscillation
        should only be considered over the times where there are bursts.

    Returns
    -------
    signal : 1d array
        Bursty oscillation with noise time series.
    oscillation : 1d array
        Bursty oscillation component of signal.
    noise : 1d array
        Noise component of signal.
    df : pd.DataFrame
        Cycle-by-cycle properties of the simulated oscillation.
    """

    # Generate & then demean noise
    noise = _return_noise_sim(n_seconds, fs, noise_generator, noise_args)
    noise = noise - noise.mean()

    # Generate oscillation
    oscillation, df = sim_bursty_oscillation(n_seconds, fs, freq, rdsym=rdsym,
                                             prob_enter_burst=prob_enter_burst,
                                             prob_leave_burst=prob_leave_burst,
                                             cycle_features=cycle_features,
                                             return_cycle_df=True)

    # Determine samples of burst so that can compute signal power over only those times
    is_osc = np.zeros(len(oscillation), dtype=bool)
    for ind, row in df.iterrows():
        if row['is_cycle']:
            is_osc[row['start_sample']:row['start_sample'] + row['period']] = True

    # Normalize & combine signal
    oscillation, noise = normalize_by_variance(oscillation, noise, ratio_osc_var)
    signal = oscillation + noise

    if return_components:
        if return_cycle_df:
            return signal, oscillation, noise, df
        return signal, oscillation, noise
    else:
        if return_cycle_df:
            return signal, df
        return signal


def normalize_by_variance(sig1, sig2, ratio):
    """Normalize the variance across two signals.

    Parameters
    ----------
    sig1, sig2 : 1d array
        Vectors of data to normalize variance with respect to each other.
    ratio : float
        Desired ratio of sig1 variance to sig2 variance.
        If >1 - sig1 is stronger, if <1 - noise is stronger.

    Returns
    -------
    1d array, 1d array
        Vectors of data, where sig2 has been normalized so the desired variance ratio is attained.
    """

    return sig1, np.sqrt(sig2**2 * np.var(sig1) / (np.var(sig2) * ratio)) * np.sign(sig2)
