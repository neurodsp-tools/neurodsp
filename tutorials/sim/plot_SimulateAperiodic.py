"""
Simulating Aperiodic Signals
============================

Simulate aperiodic signals.

This tutorial covers the the :mod:`~neurodsp.sim.aperiodic` module.
"""

###################################################################################################

# Import sim module
from neurodsp.sim import set_random_seed, sim_powerlaw, sim_random_walk, sim_synaptic_current

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
n_seconds = 10

###################################################################################################
# Simulate 1/f Activity
# ---------------------
#
# Often, we want to simulate noise that is comparable to what we see in neural recordings.
#
# Neural signals display 1/f-like activity, whereby power decreases linearly across
# increasing frequencies, when plotted in log-log.
#
# To simulate activity with powerlaw distributions, use the
# :func:`~.sim_powerlaw` function.
#
# Let's start with a power law signal, specifically a brown noise process, or a signal
# for which the power spectrum is distributed as 1/f^2.
#

###################################################################################################

# Setting for the simulation
exponent = -2

# Simulate powerlaw activity, specifically brown noise
times = create_times(n_seconds, fs)
br_noise = sim_powerlaw(n_seconds, fs, exponent)

###################################################################################################

# Plot the simulated data, in the time domain
plot_time_series(times, br_noise)

###################################################################################################

# Plot the simulated data, in the frequency domain
freqs, psd = compute_spectrum(br_noise, fs)
plot_power_spectra(freqs, psd)

###################################################################################################
# Simulate Filtered 1/f Activity
# ------------------------------
#
# The power law simulation function is also integrated with a filter. This can be useful
# for filtering out some low frequencies, as is often done with neural signals,
# to remove the very slow drifts that we see in the pure 1/f simulations.
#
# To filter a simulated power law signal, simply pass in a filter range, and the filter will
# be applied to the simulated data before being returned. Here we will apply a high-pass filter.
#
# We can see that the resulting signal has much less low-frequency drift than the first one.
#

###################################################################################################

# Simulate highpass-filtered brown noise with a 1Hz cutoff frequency
f_hipass_brown = 1
brown_filt = sim_powerlaw(n_seconds, fs, exponent, f_range=(f_hipass_brown, None))

###################################################################################################

# Plot the simulated data, in the time domain
plot_time_series(times, brown_filt)

###################################################################################################

# Plot the simulated data, in the frequency domain
freqs, psd = compute_spectrum(brown_filt, fs)
plot_power_spectra(freqs, psd)

###################################################################################################
#
# Note: the :func:`~.sim_powerlaw` function can simulate arbitrary
# power law exponents, such as pink noise (-1), or any other exponent.
#

###################################################################################################
# Random Walk Activity
# --------------------
#
# We can also simulate an Ornstein-Uhlenbeck process, which is a random walk process with memory.
#
# We can do this with the :func:`~.sim_random_walk` function.
#

###################################################################################################

# Simulate aperiodic signals from a random walk process
rw_noise = sim_random_walk(n_seconds, fs)

###################################################################################################

# Plot the simulated data, in the time domain
plot_time_series(times, rw_noise, title='RW Process')

###################################################################################################
# Simulate Synaptic Activity
# --------------------------
#
# Another model for simulating aperiodic, neurally plausible activity, is to simulate
# synaptic current activity, as a Lorentzian function.
#
# This is available with the :func:`~.sim_synaptic_current` function.
#
# The synaptic current model is Poisson activity convolved with exponential kernels
# that mimic the shape of post-synaptic potentials.
#
# For more details on the usage of such models for simulating neural signals,
# see Destexhe et al., 2001 and/or Gao et al., 2017.
#

###################################################################################################

# Simulate aperiodic activity from the synaptic kernel model
syn_noise = sim_synaptic_current(n_seconds, fs)

###################################################################################################

# Plot the simulated data, in the time domain
plot_time_series(times, syn_noise, title='Synaptic Activity')

###################################################################################################
#
# Both the random walk, and synaptic model produce 1/f scaling in higher frequencies with a
# fixed exponent of -2, as we can see in the power spectra plot below.
#

###################################################################################################

# Plot the simulated data, in the frequency domain
freqs, rw_psd = compute_spectrum(rw_noise, fs)
freqs, syn_psd = compute_spectrum(syn_noise, fs)

plot_power_spectra(freqs, [rw_psd, syn_psd], ['RW', 'Synaptic'])

###################################################################################################
#
# Sphinx settings:
# sphinx_gallery_thumbnail_number = 3
#
