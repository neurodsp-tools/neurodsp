"""
Morlet Wavelet Analysis
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
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import simulation code to create test Data
from neurodsp.sim import sim_bursty_oscillation
from neurodsp.utils import set_random_seed, create_times

# Import functions for morlet Wavelets
from neurodsp.timefrequency.wavelets import compute_wavelet_transform

###################################################################################################

# Set the random seed, for consistency simulating data
set_random_seed(0)

###################################################################################################
# Simulate time-frequency series data
# -----------------------------------
#
# First, we'll simulate the time frequency series using the function ':func:sim_bursty_oscillation'.
# This will simulate time-varying oscillations.
# For this example, our oscillation frequency will be 20 Hz, with a sampling rate of 500 s, and a
# simulation time of 10 seconds.

###################################################################################################

# Define keyword arguments for bursty oscillation function
fs = 500
n_seconds = 10
freq = 20

# Simulate a signal with bursty oscillations at 20 Hz
sig = sim_bursty_oscillation(n_seconds, fs, freq)
times = create_times(n_seconds, fs)

###################################################################################################
# Compute Wavelet Transform Algorithm
# ----------------------------------
#
# Now, lets use the compute Morlet wavlet transform algorithm to transform our simulated time-series
# data to a time-frequency representation using Morlet wavelets.
# The algorithm computes the continuous Morlet wavelet transform at the specified frequencies and
# across all shifts.
# For this example, we'll compute the Morlet wavelet transform on 50 equally-spaced frequencies
# from 5 Hz to 100 Hz.

###################################################################################################

# Settings for the wavelet transform algorithm
freqs = np.linspace(5, 100, 50)

# Compute wavelet transform using compute morlet wavelet transform algorithm
mwt = compute_wavelet_transform(sig, fs=fs, n_cycles=7, freqs=freqs)

###################################################################################################

# Plot morlet wavelet transform
fig, ax = plt.subplots()
ax.imshow(abs(mwt), aspect='auto')
ax.invert_yaxis()
ax.set_xlabel('time (s)')
ax.set_xticks(np.linspace(0, times.size, 5))
ax.set_xticklabels(np.round(np.linspace(times[0], times[-1], 5), 2))
ax.set_ylabel('freq (Hz)')
ax.set_yticks(np.linspace(0, freqs.size, 5))
ax.set_yticklabels(np.round(np.linspace(freqs[0], freqs[-1], 5), 2))
fig.show()

###################################################################################################
#
# From the plot above, you can see the Morlet-wavelet transformed signal.
#

###################################################################################################
#
# We can also change the frequencies passed to the morlet-wavelet transform algorithm.
# For example, let's use an array of frequencies from 15 Hz to 50 Hz with a spacing of 5 Hz.

###################################################################################################

# Settings for the wavelet transform algorithm
freqs = np.arange(15, 50, 5)

# Compute wavelet transform using compute morlet wavelet transform algorithm
mwt = compute_wavelet_transform(sig, fs=fs, n_cycles=7, freqs=freqs)

###################################################################################################

# Plot morlet wavelet transform
fig, ax = plt.subplots()
ax.imshow(abs(mwt), aspect='auto')
ax.invert_yaxis()
ax.set_xlabel('time (s)')
ax.set_xticks(np.linspace(0, times.size, 5))
ax.set_xticklabels(np.round(np.linspace(times[0], times[-1], 5), 2))
ax.set_ylabel('freq (Hz)')
ax.set_yticks(np.linspace(0, freqs.size, 5))
ax.set_yticklabels(np.round(np.linspace(freqs[0], freqs[-1], 5), 2))
fig.show()

###################################################################################################
#
# From the plot above, you can see the Morlet-wavelet transformed signal for the new frequency range.
#

###################################################################################################
#
# It is useful to describe the parameters needed to define a Morlet wavelet. These parameters are
# the sampling frequency, the fundamental frequency, and the number of cycles per frequency.
#
# It is also usefule to describe what the Morlet wavelet is in the context of this tutorial.
# Simply put, a morlet wavelet takes a raw input signal and multiplies it by a Gaussian envelope.
# It is useful in cases where a signal's amplitude varies over time.
#
# For more information on Morlet wavelets, see:
# Mike X Cohen, 2019, "Morlet wavelets in time and frequency,"
# YouTube, https://www.youtube.com/watch?v=7ahrcB5HL0k.
#

###################################################################################################

# Example plot of morlet wavelet
# ------------------------------
#
# Here, I provide an example plot of a Morlet wavelet to demonstrate my previous description

###################################################################################################

# Use the scipy.signal function 'morlet' to create a wavelet.
# Here, we use a length of 175 with a total of 7 cycles.

# Define sampling rate, number of cycles, fundamental frequency, and length for the wavelet.
fs = 500
n_cycles = 7
freq = 5
s = 1.0
w = n_cycles
M = int(n_cycles * fs / freq)

# Create wavelet
wavelet = signal.morlet(M, w, s)

# Plot wavelet
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.linspace(0, s, wavelet.size), wavelet.real, wavelet.imag)
fig.show()

###################################################################################################
#
# From the plot above, you can see the Morlet-wavelet.
# The y-axis shows the amplitude of the signal, and the z-axis shows the rotation.
#
# Adjusting the input parameters results in a different image.
# For example, let's try this same plot but with a different number of cycles:

###################################################################################################

fs = 500
n_cycles = 15
freq = 5
s = 1.0
w = n_cycles
M = int(n_cycles * fs / freq)

# Create wavelet
wavelet = signal.morlet(M, w, s)

# Plot wavelet
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.linspace(0, s, wavelet.size), wavelet.real, wavelet.imag)
fig.show()

###################################################################################################
#
# As you can see, when you increase the n_cycles parameter, you get more oscillations in the signal.
#
# If we return to the Morlet-wavelet transform algorithm, we can adjust input parameters
# to demonstrate how changes in the number of cycles per frequency affect our plot of the signal.

###################################################################################################

freqs = np.arange(15, 50, 5)

mwt = compute_wavelet_transform(sig, fs=fs, n_cycles=15, freqs=freqs)

fig, ax = plt.subplots()
ax.imshow(abs(mwt), aspect='auto')
ax.invert_yaxis()
ax.set_xlabel('time (s)')
ax.set_xticks(np.linspace(0, times.size, 5))
ax.set_xticklabels(np.round(np.linspace(times[0], times[-1], 5), 2))
ax.set_ylabel('freq (Hz)')
ax.set_yticks(np.linspace(0, freqs.size, 5))
ax.set_yticklabels(np.round(np.linspace(freqs[0], freqs[-1], 5), 2))
fig.show()

###################################################################################################
#
# As you can see, increasing n_cycles results in a higher number of oscillations
# within the time-frequency domain.
#
# If we adjust other input parameters, such as the frequency range, we can also get a different result.

###################################################################################################

# Here, we made the range of frequencies passed to the algorithm smaller.
freqs = np.arange(15, 50, 10)

mwt = compute_wavelet_transform(sig, fs=fs, n_cycles=15, freqs=freqs)

fig, ax = plt.subplots()
ax.imshow(abs(mwt), aspect='auto')
ax.invert_yaxis()
ax.set_xlabel('time (s)')
ax.set_xticks(np.linspace(0, times.size, 5))
ax.set_xticklabels(np.round(np.linspace(times[0], times[-1], 5), 2))
ax.set_ylabel('freq (Hz)')
ax.set_yticks(np.linspace(0, freqs.size, 5))
ax.set_yticklabels(np.round(np.linspace(freqs[0], freqs[-1], 5), 2))
fig.show()

###################################################################################################
#
# From this plot, you can see that with a larger frequency step,
# with the same starting and ending frequencies, the amplitude of the estimated frequencies is
# larger in the time-frequency domain.

###################################################################################################
