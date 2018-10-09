from .version import __version__

from .filt import filter
from .laggedcoherence import lagged_coherence
from .timefrequency import phase_by_time, amp_by_time, freq_by_time
from .burst import detect_bursts, detect_bursts_bosc
from .sim import sim_filtered_brown_noise, sim_brown_noise, sim_oscillator, sim_noisy_oscillator, sim_bursty_oscillator, sim_noisy_bursty_oscillator, sim_synaptic_noise, sim_OU_process, sim_poisson_pop, sim_variable_powerlaw
from .swm import sliding_window_matching
from . import spectral
