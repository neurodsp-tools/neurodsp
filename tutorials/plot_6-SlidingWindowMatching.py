"""
Sliding Window Matching
=======================

Find recurrent patterns in a neural signal using Sliding Window Matching.

This tutorial primarily covers :mod:`neurodsp.swm`.
"""

###################################################################################################
#
# Overview
# --------
#
# This notebook shows how to implement sliding window matching (SWM) for
# identifying recurring patterns in a neural signal, like the shape of an
# oscillatory waveform.
#
# For more details on Sliding Window Matching see Gips et al., 2017, J Neuro Methods.
#

###################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from neurodsp.swm import sliding_window_matching

###################################################################################################

# Set the random seed, for consistency simulating data
np.random.seed(0)

###################################################################################################
#
# Load neural signal
# ------------------
#

###################################################################################################

# Load example data
sig = np.load('./data/sample_data_1.npy')
fs = 1000
times = np.arange(0, len(sig)/fs, 1/fs)
f_range = (13, 30)

# Plot example signal
plt.figure(figsize=(12, 3))
plt.plot(times, sig, 'k')
plt.xlim((4, 5))
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')
plt.tight_layout()

###################################################################################################
#
# Apply sliding window matching to neural signal
# ----------------------------------------------
#
# Because we define the window length to be about 1 cycle, this should
# roughly extract the waveform shape of the neural oscillation. Notice
# that the beta cycles have sharper troughs than peaks, and the average
# window is a beta cycle with a sharp trough.
#
# However, notice that these results change dramatically by changing the
# random seed. Using more data and increasing the number of iterations
# would help the robustness of the algorithm.
#

###################################################################################################

# Define window length & minimum window spacing, both in seconds
win_len = .055
win_spacing = .2

# Apply the sliding window matching algorithm to the time series
avg_window, window_starts, J = sliding_window_matching(sig, fs, win_len, win_spacing,
                                                       max_iterations=500)

###################################################################################################

# Plot the discovered pattern
plt.figure(figsize=(4, 4))
plt.plot(avg_window, 'k')
plt.xlabel('Time (samples)')
plt.ylabel('Voltage (a.u.)')
plt.title('Average pattern in neural signal')
plt.tight_layout()

###################################################################################################
#
# Sphinx settings:
# sphinx_gallery_thumbnail_number = 2
#
