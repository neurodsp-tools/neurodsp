"""Simulating time series, with combinations of periodic, aperiodic and transient components."""

import numpy as np

from neurodsp.sim.periodic import sim_oscillator, sim_bursty_oscillator
from neurodsp.sim.aperiodic import _return_noise_sim

###################################################################################################
###################################################################################################

def sim_noisy_oscillator(n_seconds, fs, freq, noise_generator='synaptic',
                         noise_args={}, rdsym=.5, ratio_osc_var=1):
    """Simulate an oscillation embedded in background 1/f noise.

    Parameters
    ----------
    n_seconds : float
        Signal duration, in seconds.
    fs : float
        Signal sampling rate, in Hz.
    freq : float
        Oscillator frequency.
    noise_generator: str or numpy.ndarray, optional, default='synaptic'
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
        Rise-decay symmetry of the oscillator, as fraction of the period in the rise time:

        - = 0.5 - symmetric (i.e., sine wave, default)
        - < 0.5 - shorter rise, longer decay
        - > 0.5 - longer rise, shorter decay
    ratio_osc_var : float, optional, default=1
        Ratio of oscillator variance to noise variance.
        If >1 - oscillator is stronger, if <1 - noise is stronger.

    Returns
    -------
    osc: 1d array
        Oscillator with noise.
    """

    # Determine length of signal in samples
    n_samples = int(n_seconds * fs)

    # Generate & demean noise
    noise = _return_noise_sim(n_seconds, fs, noise_generator, noise_args)
    noise = noise - noise.mean()

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


def sim_noisy_bursty_oscillator(n_seconds, fs, freq, noise_generator='synaptic', noise_args={},
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
        Oscillator frequency, in Hz
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
        Rise-decay symmetry of the oscillator as fraction of the period in the rise time

        - = 0.5: symmetric (sine wave)
        - < 0.5: shorter rise, longer decay
        - > 0.5: longer rise, shorter decay
    ratio_osc_var : float
        Ratio of oscillator power to noise power.
        If >1 - oscillator is stronger, if <1 - noise is stronger.
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
        Whether to return the oscillator and noise separately, in addition to the signal.
    return_cycle_df : bool
        If True, return the dataframe that contains the simulation parameters for each cycle.
        This may be useful for computing power, for example, as the power of the oscillator
        should only be considered over the times where there are bursts.

    Returns
    -------
    signal : 1d array
        Bursty oscillator with noise time series.
    oscillator : 1d array
        Bursty oscillator component of signal.
    noise : 1d array
        Noise component of signal.
    df : pd.DataFrame
        Cycle-by-cycle properties of the simulated oscillator.
    """

    # Generate & then demean noise
    noise = _return_noise_sim(n_seconds, fs, noise_generator, noise_args)
    noise = noise - noise.mean()

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
