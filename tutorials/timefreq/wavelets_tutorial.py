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
from neurodsp.filt.filter import filter_signal

# Import simulation code to create test Data
from neurodsp.sim import sim_combined, sim_bursty_oscillation
from neurodsp.sim.cycles import sim_cycle
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

# Define filter key word arguments, and filter_signal. Let's use a lowpass filter with a high frequency cutoff at 14 Hz.
f_hi = 14
pass_type = 'lowpass'
filter_kwargs = {'pass_type':pass_type}

# Define simulation components
components = {'sim_powerlaw': {'exponent':-2.0, **filter_kwargs}, 'sim_oscillation' : {'freq':10}}

# Simulate a signal with a power-law time series with oscillations at 10 Hz.
sig = sim_combined(n_seconds, fs, components)
times = create_times(n_seconds, fs)

###################################################################################################

# Plot the simulated data
plot_time_series(times, sig, 'Simulated EEG')
plt.show()

###################################################################################################
#
# In the simulated signal above, we can see the time series data.
#The time ranges from 0 to 10 seconds, and the voltage oscillates between -2 and 2 microvolts.
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
mwt = compute_wavelet_transform(sig, fs=500, freqs=freqs)

###################################################################################################
# Plot morlet wavelet transform
# You can plot the wavelet-transformed data using matplotlib:

plt.imshow(abs(mwt), aspect='auto')
plt.show()

###################################################################################################
#
# The plot above shows the morlet-wavelet transformation of the simulated signal
# using a lowpass filter with a high-frequency cutoff at 14 Hz, and oscillations at 10 Hz.
# You can change the parameters of the filter and of the simulated signal and still apply the
# morlet-wavelet transformation algorithm. We can do this be changing the filter keyword arguments,
# and by using a different algorithm to simulate oscillations.
# For example, let's use a highpass filter with a low frequency cutoff at 5 Hz, and simulate some time-varying oscillations.
#

###################################################################################################
# To simulate the time-varying oscillations, we will use the function 'sim_bursty_oscillation', which can be found under neurodsp.periodic.py.
# It takes input parameters n_seconds, fs, freq, enter_burst, leave_burst, and cycle.
# The n_seconds parameter is the simulation time.
# The fs parameter is the sampling rate of the simulated signal.
# The freq parameter is the oscillation frequency.
# The enter_burst parameter is the probability of a cycle being oscillating given the last cycle is not oscillating. The default argument is .2.
# The leave_burst parameter is the probability of a cycle not being oscillating given the last cycle is oscillating. The default argument is .2.
# The cycle parameter is the type of oscillation cycle being simulated, with options including 'sine', 'asine', 'sawtooth', 'gaussian', 'exp', and '2exp'.

# For this example, let's use a cycle with exponential decay. The 'exp' parameter takes a key word argument of 'tau_d', which specifies the decay time.
# We will use a decay time of 2 seconds. Our oscillation frequency will be 20 Hz, with a sampling rate of 500 s, and a simulation time of 10 seconds.

# For our filter keyword arguments, our highpass filter will have a low frequency cutoff will be at 5 Hz.

# Simulation settings
fs = 500
n_seconds = 10
freq = 20

# Filter key word arguments
f_lo = 5
pass_type = 'highpass'
filter_kwargs = {'pass_type':pass_type}

# Cycle key word arguments
tau_d = 2
cycle = 'exp'
cycle_kwargs = {'cycle':cycle, 'tau_d':tau_d}

# Simulate a signal with bursty oscillations at 20 Hz with a decay time of 2 seconds
sig = sim_bursty_oscillation(n_seconds, fs, freq, **cycle_kwargs)
times = create_times(n_seconds, fs)

###################################################################################################

# Plot the simulated data
plot_time_series(times, sig, 'Simulated EEG')
plt.show()

###################################################################################################
#
# In the simulated signal above, we can see the time series data for our signal.
#
###################################################################################################

# Now, let's again apply our wavelet-transform algorithm.

# Settings for the wavelet transform Algorithm.
#
# The frequencies can be specified as a 1D array, or as a list. If
# specified as an array, the frequency inputs inform which frequency values to estimate with Morlet wavelets.
# If specified as a list, the frequency inputs define the frequency range over which to estimate with wavelets.
#
# I will do an example of both below.

# Estimate frequencies 6 Hz and 40 Hz with Morlet Wavelets:
freqs_1= [6, 40]

# Estimate range of frequencies from 15 Hz to 30 Hz:
freqs_2= (15, 30)

# Compute wavelet transform using compute morlet wavelet transform algorithm
mwt_1 = compute_wavelet_transform(sig, fs=500, n_cycles=7, freqs=freqs_1)
mwt_2 = compute_wavelet_transform(sig, fs=500, n_cycles=7, freqs=freqs_2)

###################################################################################################
# Plot morlet wavelet transform

# For estimated frequencies 6 Hz and 40 Hz:
plt.imshow(abs(mwt_1), aspect='auto')
plt.show()

# For estimated range of frequencies from 15 Hz to 30 Hz:
plt.imshow(abs(mwt_2), aspect='auto')
plt.show()

###################################################################################################
#
# The plot above shows the morlet-wavelet transformation of the simulated signal
# using a highpass filter with a low-frequency cutoff at 5 Hz, and time-varying oscillations at 20 Hz with a decay time of 2 s.
#

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
# Plot the analytic amplitude and the analytic phase as functions of time for the convolved wavelet.

# Let's look at the analytic amplitude and the analytic phase of the convolved signal.
analytic_amp = np.abs(cts)
analytic_phase = np.angle(cts)

# Looking at analytic_amp, you can see that the returned analytic amplitude is an array of 5000 elements,
# and looking at analytic_phase, you can see that the returned analytic phase is also an array of 5000 elements.
# Let's look at the analytic amplitude vs. time, and the analytic phase vs. time.

# For the analytic amplitude vs. time:

plt.imshow((times, analytic_amp), aspect='auto')
plt.show()

# For the analytic phase vs. time:

plt.imshow((times, analytic_phase), aspect='auto')
plt.show()
