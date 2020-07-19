"""
Simulating Periodic Signals
===========================

Simulate periodic, or oscillatory, signals.

This tutorial covers the ``neurodsp.sim.periodic`` module.
"""

###################################################################################################

# sphinx_gallery_thumbnail_number = 1

# Import sim module
from neurodsp.sim import set_random_seed, sim_oscillation, sim_bursty_oscillation

# Import function to compute power spectra
from neurodsp.spectral import compute_spectrum

# Import utilities for plotting data
from neurodsp.utils import create_times
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series

###################################################################################################

# Set the random seed, for consistency simulating data
set_random_seed(0)

# Set some general settings, to be used across all simulations
fs = 1000

###################################################################################################
# Simulate a Stationary Oscillation
# ---------------------------------
#
# Let's start by simulating an oscillation. We'll start with a simple, sinusoidal, oscillation.
#
# Continuous periodic signals can be created with :func:`~.sim_oscillation`.
#

###################################################################################################

# Simulation settings
n_seconds = 1
osc_freq = 6.6

# Simulate a sinusoidal oscillation
osc_sine = sim_oscillation(n_seconds, fs, osc_freq, cycle='sine')

# Create a times vector for our simulation
times = create_times(n_seconds, fs)

# Plot the simulated data, in the time domain
plot_time_series(times, osc_sine)

###################################################################################################
# Cycle Kernels
# -------------
#
# To simulate oscillations, we can use a sinusoidal kernel, as above, or any of a
# selection of other cycle kernels.
#
# Different kernels represent different shapes and properties that may be useful to
# simulate different aspects of periodic neural activity.
#
# Cycle kernel options include:
#
# - ``sine``: a sine wave cycle
# - ``asine``: an asymmetric sine wave
# - ``sawtooth``: a sawtooth wave
# - ``gaussian``: a gaussian cycle
# - ``exp``: a cycle with exponential decay
# - ``2exp``: a cycle with exponential rise and decay
#
# Note that these cycle kernels are all created with the
# :func:`~.sim_cycle` function.
#

###################################################################################################
# Simulate a Shapely Oscillation
# ------------------------------
#
# Next let's simulate an asymmetric oscillation, using the `asine` cycle kernel,
# which stands for 'asymmetric sinusoidal'.
#
# Using the `asine` kernel, we can simulate arbitrary rise-decay symmetry of oscillations.
#
# We'll plot it over our original sinusoidal oscillation, so we can compare them.
#

###################################################################################################

# Simulate a shape-y oscillations
rdsym = 0.2
osc_shape = sim_oscillation(n_seconds, fs, osc_freq,
                            cycle='asine', rdsym=rdsym)

###################################################################################################

# Plot the simulated data, in the time domain
plot_time_series(times, [osc_sine, osc_shape],
                 labels=['rdsym='+str(.5), 'rdsym='+str(rdsym)])

###################################################################################################
#
# We can also compare these signals in the frequency domain.
#
# Notice that the asymmetric oscillation has strong harmonics resulting from the
# non-sinusoidal nature of the oscillation.
#

###################################################################################################

# Plot the simulated data, in the frequency domain
freqs_sine, psd_sine = compute_spectrum(osc_sine, fs)
freqs_shape, psd_shape = compute_spectrum(osc_shape, fs)

plot_power_spectra([freqs_sine, freqs_shape], [psd_sine, psd_shape])

###################################################################################################
# Simulate a Bursty Oscillation
# -----------------------------
#
# Sometimes we want to study oscillations that come and go, so it can be useful to simulate
# oscillations with this property.
#
# You can simulate bursty oscillations with :func:`~.sim_bursty_oscillation`.
#
# To control the bursty-ness of the simulated signal, you can control the probability
# that a burst will start or stop with each new cycle.
#

###################################################################################################

# Settings for simulation
osc_freq = 30
n_seconds = 3
enter_burst = .1
leave_burst = .1

# Simulate a bursty oscillation
osc = sim_bursty_oscillation(n_seconds, fs, osc_freq,
                             enter_burst=enter_burst,
                             leave_burst=leave_burst)
times = create_times(n_seconds, fs)

###################################################################################################

# Plot the simulated data, in the time domain
plot_time_series(times, osc, xlim=[0, n_seconds])

###################################################################################################
#
# We can shorten burst duration by increasing the probability to leave bursts.
#

###################################################################################################

# Simulate a bursty oscillation, with a specified burst probability
leave_burst = .4
osc = sim_bursty_oscillation(n_seconds, fs, osc_freq,
                             enter_burst=enter_burst,
                             leave_burst=leave_burst)
times = create_times(n_seconds, fs)

###################################################################################################

# Plot the simulated data, in the time domain
plot_time_series(times, osc, xlim=[0, n_seconds])

###################################################################################################
#
# We can increase the number of bursts by increasing the probability to enter a burst.
#

###################################################################################################

# Simulate a bursty oscillation, with a specified burst probability
enter_burst = .4
osc = sim_bursty_oscillation(n_seconds, fs, osc_freq,
                             enter_burst=enter_burst,
                             leave_burst=leave_burst)
times = create_times(n_seconds, fs)

###################################################################################################

# Plot the simulated data, in the time domain
plot_time_series(times, osc, xlim=[0, n_seconds])
