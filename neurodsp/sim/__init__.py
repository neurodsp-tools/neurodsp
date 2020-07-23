"""Simulate neural time series, with periodic, aperiodic, and transient components."""

from neurodsp.utils.sim import set_random_seed

from .periodic import sim_oscillation, sim_bursty_oscillation
from .aperiodic import sim_powerlaw, sim_random_walk, sim_synaptic_current, sim_poisson_pop
from .cycles import sim_cycle
from .transients import sim_synaptic_kernel
from .combined import sim_combined
