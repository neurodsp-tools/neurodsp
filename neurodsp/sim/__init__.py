"""Simulate neural time series, with periodic, aperiodic, and transient components."""

from neurodsp.utils.sim import set_random_seed

from .periodic import (sim_oscillation, sim_bursty_oscillation,
                       sim_variable_oscillation, sim_damped_oscillation)
from .aperiodic import (sim_powerlaw, sim_random_walk, sim_synaptic_current, sim_poisson_pop,
                        sim_knee, sim_frac_gaussian_noise, sim_frac_brownian_motion)
from .cycles import sim_cycle
from .transients import sim_synaptic_kernel, sim_action_potential
from .combined import sim_combined, sim_peak_oscillation, sim_modulated_signal
