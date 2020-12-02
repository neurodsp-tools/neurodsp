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
# First, we'll simulate the time frequency series using the function 'sim_bursty_oscillation'. This will simulate time-varying oscillations.
# For this example, our oscillation frequency will be 20 Hz, with a sampling rate of 500 s, and a simulation time of 10 seconds.

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
# Now, lets use the Compute Morlet Wavlet Transform algorithm to transform our simulated time-series Data
# to a time-frequency representation using morlet wavelets.
# The algorithm computes the continuous morlet wavelet transform at the specified frequencies and across all shifts.
# For this example, we'll compute the Morlet wavelet transform on 50 equally spaced frequencies from 5 Hz to 100 Hz.

###################################################################################################

# Settings for the wavelet transform Algorithm
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
