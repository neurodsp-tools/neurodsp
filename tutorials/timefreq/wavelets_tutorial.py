"""
Wavelets
===============

Perform time-frequency decomposition using wavelets.

This tutorial will explore the ``neurodsp.timefrequency.wavelets`` module.
It uses Morlet wavelets to transform time series data to a time-frequency
representation of the data.
"""

###################################################################################################

# Import neccessary functions and packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import morlet

from neurodsp.utils.data import create_freqs
from neurodsp.utils.checks import check_n_cycles
from neurodsp.utils.decorators import multidim

# Import simulation code to create test Data
from neurodsp.sim import sim_combined
from neurodsp.utils import set_random_seed, create_times

# Import utilities for loading and plotting data
from neurodsp.utils.download import load_ndsp_data
from neurodsp.plts.time_series import plot_time_series

# Import functions for morlet Wavelets
from neurodsp.timefrequency.wavelets import compute_wavelet_transform, convolve_wavelet

###################################################################################################

# Set the random seed, for consistency simulating data
set_random_seed(0)

###################################################################################################
# Simulate time-frequency series data
# -----------------------------
#
# First, we'll simulate the time frequency series using the function
# 'sim_combined', which combines multiple component signals to simulate a signal.
# It takes input parameter n_seconds and fs, which specify the simulation time and signal
# sampling rate, respectively.
# It also takes components which include 'sim_powerlaw',
# 'sim_oscillation', and 'freq'.
# These components must be specified separately.
#
# The 'sim_powerlaw' function can be found under neurodsp.aperiodic.py.
# It simulates a power law time series with a specified exponent. We'll also need
# the key word arguments to pass to the function filter_signal, which can be found under
# neurodsp.filt.filter.py.
#
# The filter_signal function has input parameters sig, fs, pass_type, f_range,
# filter_type, n_cycles, n_seconds, remove_edges, butterworth_order,
# print_transitions, plot_properties, and return_filter.
# The sig parameter defines the time series to be filtered.
# The fs parameter is the sampling rate.
# The pass_type parameter specifies the type of filter to apply. The options include
# bandpass, bandstop, lowpass, and highpass.
# The f_range parameter defines the cutoff frequency or frequencies used for the filter,
# specified as f_lo and f_hi.
# The filter_type parameter is optional and determines whether to use a FIR or IIR
# filter (the only IIR filter option is a butterworth filter).
# The n_cycles parameter is the length of the filter, in number of cycles, at the 'f_lo'
# frequency if using a FIR filter.
# The n_seconds parameter is the length of the filter, in seconds, if using a FIR filter.
# The remove_edges parameter is an optional boolean parameter. If set to True, it replaces
# samples within half the kernel length to be np.nan (only for FIR filters).
# The butterworth_order parameter is the order of the butterworth filter, if using
# an IIR filter.
#
# For this tutorial, we will use a lowpass filter FIR filter.
#
# The 'sim_oscillation' function can be found under neurodsp.periodic.py.
# It simulates an oscillation, and takes input parameters n_seconds, fs, freq,
# cycle, phase, and cycle_params.
# The cycle is specified as sine, and the phase default is 0.
# The cycle_params are the parameters for the simulated oscillation cycle,
# which are already stored.
#
# We'll also need the function 'create_times' to create an array of time indices.
# It takes input parameters n_seconds, fs, and start_val.
# The n_seconds parameter is the length of the signal.
# The fs parameter is the sampling frequency.
# The start_val parameter is the starting value for the time definition, and it
# has a default value of 0.

###################################################################################################

# Simulation settings
fs = 500
n_seconds = 10

# Define simulation components
components = {'sim_powerlaw': {'exponent' : {-2.0}}, {'f_range' : {None}}, {'**filter_kwargs' : {'sig' : {'sim_combined' : {n_seconds, fs}}, {'pass_type': {'lowpass'}}}}, 'sim_oscillation' : {'freq': 10}})

# Simulate a signal with a power-law time series with oscillations at 10 Hz.
sig = sim_combined(n_seconds, fs, components)
times = create_times(n_seconds, fs)

###################################################################################################

# Plot the simulated data
plot_time_series(times, sig, 'Simulated EEG')

###################################################################################################
#
# In the simulated signal above, we can see the time series data.
#

###################################################################################################
# Compute Wavelet Transform Algorithm
# ----------------------------------
#
# Now, lets use the Compute Morlet Wavlet Transform algorithm to transform our simulated time-series Data
# to a time-frequency representation using morlet wavelets.
# The algorithm computes the continuous morlet wavelet transform
# at the specified frequencies and across all shifts.
#
# It takes parameters sig, fs, freqs, n_cycles, scaling, and norm.
# The sig parameters is the time series signal.
# The fs parameter is the sampling rate.
# The freqs parameter defines the frequencies values or frequency range to estimate
# with morlet wavelets.
# The n_cycles parameter defines the number of cycles for each frequency.
# The scaling parameter is the scaling factor for the morlet transform algorithm.
# The norm parameter is optional, and defines the normalization method for the algorithm.
# It can be specified as 'sss', or as 'amp'.
# Specifying 'sss' divides by the square root of the sum of the squares.
# Specifying 'amp' divides by the sum of the amplitudes.
#
###################################################################################################

# Settings for the wavelet transform Algorithm
freqs=[1, 30]

# Compute wavelet transform using compute morlet wavelet transform algorithm
mwt = compute_wavelet_transform(fs=500, sig, freqs)

###################################################################################################
# Plot morlet wavelet transform
# You can plot the wavelet-transformed data using matplotlib:

plt.imshow(abs(mwt), aspect='auto')
plt.show()


###################################################################################################
# Convolve Wavelet Algorithm
# ----------------------------------
#
# Now, let's use the Convolve Wavelet algorithm to convolve a signal with a Complex
# morlet wavelet. This is very similar to the morlet transform algorithm, but the
# convolve wavelet algorithm returns a complex time series as an array.
# The real part of this array is the filtered signal. You can also take np.abs()
# of the array to get the analytic amplitude. Similarly, you can take np.angle()
# of the returned array to get the analytic phase.
#
# The Convolve Wavelet algorithm takes parameters sig, fs, freqs, n_cycles,
# scaling, wavelet_len, and norm.
# The sig parameters is the time series signal.
# The fs parameter is the sampling rate.
# The freqs parameter defines the frequencies values or frequency range to estimate
# with morlet wavelets.
# The n_cycles parameter defines the number of cycles of the oscillation with the
# specified frequency.
# The scaling parameter is the scaling factor for the morlet wavelet.
# The wavelet_len parameter defines the length of the wavelet. It can override the
# freq and n_cycles input parameters if so desired.
# The norm parameter is optional, and defines the normalization method for the algorithm.
# It can be specified as 'sss', or as 'amp'.
# Specifying 'sss' divides by the square root of the sum of the squares.
# Specifying 'amp' divides by the sum of the amplitudes.
#
###################################################################################################

# Settings for the convolve wavelet Algorithm
freq=10

# Convolve a signal with a complex morlet wavelet using convolve wavelet algorithm
cts = convolve_wavelet(sig, fs=500, freq=10)

###################################################################################################
# Plot convolved wavelet

# You can plot the filtered signal by plotting the real part of the convolved wavelet:
plt.imshow(mwt_real)
plt.show()
