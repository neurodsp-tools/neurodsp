"""Simulating time series, with combinations of periodic, aperiodic and transient components."""

import numpy as np

from neurodsp.sim.periodic import sim_oscillation, sim_bursty_oscillation
from neurodsp.sim.aperiodic import get_aperiodic_sim
from neurodsp.sim.utils import normalized_sum

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
    noise_generator: str, optional, default='synaptic'
        Noise model, can be one of the simulators in neurodsp.sim specificed as a string.

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
    sig : 1d array
        Combined signal with an oscillation and an aperiodic component.
    """

    # Generate & demean aperiodic signal
    aperiodic = get_aperiodic_sim(n_seconds, fs, noise_generator, noise_args)
    aperiodic = aperiodic - aperiodic.mean()

    # Generate oscillation
    oscillation = sim_oscillation(n_seconds, fs, freq, rdsym=rdsym)

    # Normalize & combine signal
    sig = normalized_sum(oscillation, aperiodic, ratio_osc_var)

    return sig


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
    noise_generator: str, optional, default: 'synaptic'
        Noise model, can be one of the simulators in neurodsp.sim specificed as a string.

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
    sig : 1d array
        Bursty oscillation with noise time series.
    oscillation : 1d array
        Bursty oscillation component of signal.
    noise : 1d array
        Noise component of signal.
    df : pd.DataFrame
        Cycle-by-cycle properties of the simulated oscillation.
    """

    # Generate & demean aperiodic signal
    aperiodic = get_aperiodic_sim(n_seconds, fs, noise_generator, noise_args)
    aperiodic = aperiodic - aperiodic.mean()

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

    # NOTE: HAVE MESSED THIS UP A BIT, AS IT NOW NORMALIZES BASED ON VARIANCE INCLUDING NON BURST SAMPLES
    #   SHOULD (USED TO) HAVE: oscillation[is_osc]
    # Note: could this be fixed by adding a tag in 'normalize_variance'

    # Normalize & combine signal
    sig = normalized_sum(oscillation, aperiodic, ratio_osc_var)

    if return_components:
        if return_cycle_df:
            return sig, oscillation, aperiodic, df
        return sig, oscillation, aperiodic
    else:
        if return_cycle_df:
            return sig, df
        return sig
